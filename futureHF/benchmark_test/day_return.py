from sys import path
path.append('/work/rqiao/HFdata/cython_mew-p')
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.runner import PairRunner
from mewp.math.simple import SimpleMoving
from mewp.util.clock import Clock
from mewp.data.order import OrderType
from mewp.simulate.report import MasterReport
from mewp.simulate.report import Report
from mewp.reader.futuresqlite import SqliteReaderDce
from mewp.util.futures import get_day_db_path
from mewp.util.pair_trade_analysis import TradeAnalysis
from mewp.data.item import Contract

from joblib import Parallel, delayed
import datetime
import numpy as np
import pandas as pd
import itertools
import os
DATA_PATH = '/work/rqiao/HFdata/dockfuture'
market = 'shfe'

def get_file_size(market, contract, date):
    statinfo = os.stat(get_day_db_path(DATA_PATH, contract, date))
    return statinfo.st_size

def get_contract_list(market, contract):
    return os.listdir(DATA_PATH + '/' + market + '/' + contract)

def get_position(contract, date, DATA_PATH):
    # create a dictionary, where date is the key
    try:
        reader = SqliteReaderDce(get_day_db_path(DATA_PATH, contract, date))
        raw = reader.read_tick()
        result = int(raw['Position'].tail(1))
    except Exception:
        result = 0
    return result

def get_best_pair(date, market, contract):
    #input a date(format: '2016-01-01'), return the best pair of contract
    cont_list = get_contract_list(market, contract)
    score = []
    for i, c in enumerate(cont_list):
        score.append(get_position(c, date, DATA_PATH))
    if sum(score) == 0:
        return 0
    max_idx = np.argmax(score)
    score[max_idx] = 0
    second_max_idx = np.argmax(score)
    return (cont_list[max_idx], cont_list[second_max_idx])
#
class SpreadGuardAlgo(PairAlgoWrapper):

    # called when algo param is set
    def param_updated(self):
        # make sure parent updates its param
        super(SpreadGuardAlgo, self).param_updated()

        # algo settings

        self.min_ticksize = self.pair.x.symbol.min_ticksize
        # create rolling
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])

        self.spreadx_roll = SimpleMoving(size = self.param['rolling'])
        self.spready_roll = SimpleMoving(size = self.param['rolling'])

        self.bollinger = self.param['bollinger']
        self.block = self.param['block']

        #other params
        self.last_long_res = -999
        self.last_short_res = -999

        #records
        self.records = {'timestamp': [], 'longs': [], 'shorts': [],
                        'long_mean': [], 'short_mean': [],
                        'long_sd': [], 'short_sd':[]}

    # what to do on every tick
    def on_tick(self, multiple, contract, info):
        # skip if price_table doesnt have both
        if self.price_table.get_size() < 2:
            return

        # get residuals and position
        long_res = self.pair.get_long_residual()
        short_res = self.pair.get_short_residual()
        pos = self.position_y()
        trade_flag = 0

        ## only do this when plotting is needed
        #update record
#       self._update_record(long_res, self.long_roll.mean, self.long_roll.sd,\
#                           short_res, self.short_roll.mean, self.short_roll.sd)

        #calculate profit for this round
        profit = 0
        if pos == -1:
            profit = long_res + self.last_short_res
        elif pos == 1:
            profit = short_res + self.last_long_res

        #two spread
        spreadx = self.spreadx_roll.mean
        spready = self.spready_roll.mean
        avg_spread = (spreadx + spready)/2

        #fee
        fee = self.pair.get_fee()

        # open or close position
        # action only when unblocked: bock size < rolling queue size
        if self.long_roll.get_size() > self.block and trade_flag == 0:
            # long when test long_res > mean+bollinger*sd
            if self.long_roll.test_sigma(long_res, self.bollinger)                    and long_res - self.long_roll.mean > fee + avg_spread:
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)
                    self.last_long_res = long_res

            # short when test short_res > mean+bollinger*sd
            elif self.short_roll.test_sigma(short_res, self.bollinger)                      and short_res - self.short_roll.mean > fee + avg_spread:
                # only short when position is 0 or 1
                if pos >= 0:
                    self.short_y(y_qty=1)
                    self.last_short_res = short_res



        # update rolling
        self.long_roll.add(long_res)
        self.short_roll.add(short_res)
        self.spreadx_roll.add(self.pair.get_spread_x())
        self.spready_roll.add(self.pair.get_spread_y())

    def on_daystart(self, date, info_x, info_y):
        # recreate rolling at each day start
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])
        self.spreadx_roll = SimpleMoving(size = self.param['rolling'])
        self.spready_roll = SimpleMoving(size = self.param['rolling'])

    def on_dayend(self, date, info_x, info_y):
        #force close on day end
        pos = self.position_y()
        # stop short position
        if pos == -1:
            self.long_y(y_qty = 1)
            return

        # stop long position
        if pos == 1:
            self.short_y(y_qty = 1)
            return

    def _update_record(self, long_res, long_mean, long_std, short_res, short_mean, short_std):
        self.records['timestamp'].append(Clock.timestamp)
        self.records['longs'].append(long_res)
        self.records['shorts'].append(short_res)
        self.records['long_mean'].append(long_mean)
        self.records['short_mean'].append(short_mean)
        self.records['long_sd'].append(long_std)
        self.records['short_sd'].append(short_std)

def back_test(pair, date, param):
    algo = { 'class': SpreadGuardAlgo }
    algo['param'] = {'x': pair[0],
                     'y': pair[1],
                     'a': 1,
                     'b': 0,
                     'rolling': param[0],
                     'bollinger': param[1],
                     'block': 100,
                     }
    settings = { 'date': date,
                 'path': DATA_PATH,
                 'tickset': 'top',
                 'algo': algo,
                 'singletick': True}
    runner = PairRunner(settings)
    runner.run()
    account = runner.account
    history = account.history.to_dataframe(account.items)
    score = float(history[['pnl']].iloc[-1])
    return score

def run_simulation(param, date_list, product):
    pnl_list = []
    for date in date_list:
        date_pair = get_best_pair(date,market, product)
        if type(date_pair) != tuple:
            continue
        else:
            pnl_list.append(back_test(date_pair, date, param))
    return pnl_list

date_list = [str(x).split(' ')[0] for x in pd.date_range('2015-01-01','2016-03-31').tolist()]
roll_list = np.arange(500, 8500, 500)
sd_list = np.arange(0.5, 4.1, 0.25)
product_list = ['ni']
pars = list(itertools.product(roll_list, sd_list))
num_cores = 20
for product in product_list:
#get trade_day_list
    trade_day_list = []
    second_contract_size_list = []
    for date in date_list:
        date_pair = get_best_pair(date, market, product)
        if type(date_pair) != tuple:
            continue
        else:
            trade_day_list.append(date)
            second_contract_size_list.append(get_file_size(market, date_pair[1], date))
    results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param,\
        date_list, product) for param in pars)
    keys = ['roll:{}_sd:{}'.format(*p) for p in pars]
    dictionary = dict(zip(keys, results))
    result = pd.DataFrame(dictionary)
    result['second_contract_size'] = second_contract_size_list
    result.index = trade_day_list
    result.to_csv('{}_day_return.csv'.format(product))
