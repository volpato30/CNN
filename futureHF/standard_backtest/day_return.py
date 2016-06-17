from sys import path
path.append('/work/rqiao/HFdata')
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
from joblib import Parallel, delayed
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pickle
import os
DATA_PATH = '/work/rqiao/HFdata/dockfuture'
market = 'shfe'

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

class Autoregressive(object):
    ## Constructor
    # @param alpha for ema
    def __init__(self, alpha):
        self.alpha = alpha

        # computed
        self.mean = 0

    ## add a new observation, and refresh
    def add(self, observe):
        if (self.mean == 0):
            self.mean = observe
        else:
            self.mean = self.mean + self.alpha * (observe - self.mean)

# Max position within 1
class StopWinAlgo(PairAlgoWrapper):

    # called when algo param is set
    def param_updated(self):
        # make sure parent updates its param
        super(StopWinAlgo, self).param_updated()

        # algo settings
        self.if_ema = self.param['if_ema'] # if false, use sma
        self.if_stop_win = self.param['if_stop_win'] #if false, don't stop win
        self.if_consider_spread = self.param['if_consider_spread'] #if false, don't consider spread and fee

        # create rolling
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])
        self.long_autoreg = Autoregressive(alpha = self.param['alpha'])
        self.short_autoreg = Autoregressive(alpha = self.param['alpha'])
        self.spreadx_roll = SimpleMoving(size = self.param['rolling'])
        self.spready_roll = SimpleMoving(size = self.param['rolling'])

        self.bollinger = self.param['bollinger']
        self.block = self.param['block']
        self.stop_win = self.param['stop_win']

        #other params
        self.last_long_res = -999
        self.last_short_res = -999

        #records
        self.records = {'timestamp': [], 'longs': [], 'shorts': [],
                        'long_mean': [], 'short_mean': [],
                        'long_sd': [], 'short_sd':[]}

        #tracker
        self.tracker = TradeAnalysis(self.pair.x)

    # what to do on every tick
    def on_tick(self, multiple, contract, info):

        self.tracker.tick_pass_by() # tell the tracker that one tick passed by
        # skip if price_table doesnt have both
        if len(self.price_table.table) < 2:
            return

        # get residuals and position
        long_res = self.pair.get_long_residual()
        short_res = self.pair.get_short_residual()
        pos = self.position_y()

        ## only do this when plotting is neede
        #update record
#         if self.if_ema:
#             self._update_record(long_res, self.autoreg.mean, self.long_roll.sd,\
#                             short_res, self.autoreg.mean, self.short_roll.sd)
#         else:
#             self._update_record(long_res, self.long_roll.mean, self.long_roll.sd,\
#                             short_res, self.short_roll.mean, self.short_roll.sd)

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

        # stop short position
        if self.if_stop_win:
            if pos == -1:
                if (profit >= max(1, self.stop_win * self.long_roll.sd) and self.if_consider_spread == False) \
                   or (profit >= max(1, self.stop_win * self.long_roll.sd, fee) and self.if_consider_spread == True):
                    self.long_y(y_qty = 1)
                    self.last_long_res = long_res
                    self.tracker.close_with_stop(profit)
                    return

            # stop long position
            if pos == 1:
                if (profit >= max(1, self.stop_win * self.long_roll.sd) and self.if_consider_spread == False) \
                   or (profit >= max(1, self.stop_win * self.long_roll.sd, fee) and self.if_consider_spread == True):
                    self.short_y(y_qty = 1)
                    self.last_short_res = short_res
                    self.tracker.close_with_stop(profit)
                    return

        # open or close position
        # action only when unblocked: bock size < rolling queue size
        if self.long_roll.queue.qsize() > self.block:
            # long when test long_res > mean+bollinger*sd
            if (long_res > self.long_autoreg.mean + self.bollinger * self.long_roll.sd \
                and self.if_ema == True and self.if_consider_spread == False) \
               or (self.long_roll.test_sigma(long_res, self.bollinger) \
                   and self.if_ema == False and self.if_consider_spread == False) \
               or (long_res - self.long_autoreg.mean > max(fee + avg_spread, self.bollinger * self.long_roll.sd) \
                   and self.if_ema == True and self.if_consider_spread == True) \
               or (self.long_roll.test_sigma(long_res, self.bollinger) \
                   and long_res - self.long_roll.mean > fee + avg_spread \
                   and self.if_ema == False and self.if_consider_spread == True): \
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)
                    self.last_long_res = long_res

                    #tell the tracker
                    if pos == 0:
                        self.tracker.open_position()
                    else:
                        self.tracker.close_with_exit(profit)

                    return

            # short when test short_res > mean+bollinger*sd
            elif (short_res > self.short_autoreg.mean + self.bollinger * self.short_roll.sd \
                  and self.if_ema == True and self.if_consider_spread == False) \
                 or (self.short_roll.test_sigma(short_res, self.bollinger) \
                     and self.if_ema == False and self.if_consider_spread == False) \
                 or (short_res - self.short_autoreg.mean > max(fee + avg_spread, self.bollinger * self.short_roll.sd) \
                     and self.if_ema == True and self.if_consider_spread == True) \
                 or (self.short_roll.test_sigma(short_res, self.bollinger) \
                     and short_res - self.short_roll.mean > fee + avg_spread \
                     and self.if_ema == False and self.if_consider_spread == True): \
                # only short when position is 0 or 1
                if pos >= 0:
                    self.short_y(y_qty=1)
                    self.last_short_res = short_res

                    #tell the tracker
                    if pos == 0:
                        self.tracker.open_position()
                    else:
                        self.tracker.close_with_exit(profit)

                    return
            else:
                pass


        # update rolling
        self.long_roll.add(long_res)
        self.short_roll.add(short_res)
        self.long_autoreg.add(long_res)
        self.short_autoreg.add(short_res)
        self.spreadx_roll.add(self.pair.get_spread_x())
        self.spready_roll.add(self.pair.get_spread_y())

    def on_daystart(self, date, info_x, info_y):
        # recreate rolling at each day start
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])
        self.long_autoreg = Autoregressive(alpha = self.param['alpha'])
        self.short_autoreg = Autoregressive(alpha = self.param['alpha'])
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
    algo = { 'class': StopWinAlgo }
    algo['param'] = {'x': pair[0],
                     'y': pair[1],
                     'a': 1,
                     'b': 0,
                     'rolling': param[0],
                     'alpha': param[1],
                     'bollinger': param[2],
                     'stop_win': param[3],
                     'block': 100,

                     'if_stop_win': True,
                     'if_ema': False,
                     'if_consider_spread': True,
                     }
    settings = { 'date': date,
                 'path': DATA_PATH,
                 'tickset': 'top',
                 'algo': algo}
    runner = PairRunner(settings)
    runner.run()
    account = runner.account
    history = account.history.to_dataframe(account.items)
    score = float(history[['pnl']].iloc[-1])
    return score

def run_simulation(param, date_list):
    pnl_list = []
    for date in date_list:
        date_pair = get_best_pair(date,market, 'al')
        if type(date_pair) != tuple:
            continue
        else:
            pnl_list.append(back_test(date_pair, date, param))
    return pnl_list

date_list = [str(x).split(' ')[0] for x in pd.date_range('2015-01-01','2016-03-31').tolist()]
roll_list = np.arange(1000, 4100, 1000)
sd_list = np.arange(1, 4.1, 0.5)
stop_win_list = np.arange(2,11)

num_cores = 20
#get trade_day_list
trade_day_list = []
for date in date_list:
    date_pair = get_best_pair(date,market, 'al')
    if type(date_pair) != tuple:
        continue
    else:
        trade_day_list.append(day)

pars = list(itertools.product(roll_list, sd_list, stop_win_list))
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param,\
    date_list) for param in pars)
keys = ['roll:{}_sd:{}_stopwin:{}'.format(*p) for p in pars]
dictionary = dict(zip(keys, results))
result = pd.DataFrame(dictionary)
result.to_csv('day_return.csv')
