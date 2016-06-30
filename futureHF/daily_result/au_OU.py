from sys import path
path.append('/work/rqiao/HFdata/cython_mew-p')
DATA_PATH = '/work/rqiao/HFdata/dockfuture'
import os
import numpy as np
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.runner import PairRunner
from mewp.simulate.report import MasterReport
from mewp.simulate.report import Report
from mewp.util.futures import get_day_db_path
from mewp.reader.futuresqlite import SqliteReaderDce
from mewp.math.simple import SimpleMoving
from mewp.util.clock import Clock
from mewp.util.pair_trade_analysis import TradeAnalysis
from mewp.data.item import Contract
import pandas as pd
from sklearn.linear_model import LinearRegression
from math import sqrt
import itertools
from joblib import Parallel, delayed
market = 'shfe'


def get_contract_list(market, contract):
    return os.listdir(DATA_PATH + '/' + market + '/' + contract)


## !!!!! always include get_day_db_path, SqliteReaderDce function
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

def get_tracker(date_list, product):
    pair = 0
    for date in date_list:
        pair = get_best_pair(date,market, product)
        if type(pair) != tuple:
            continue
        else:
            break
    return TradeAnalysis(Contract(pair[0]))

class AutoregOU(object):
    #constructor
    def __init__(self, size):
        self._observe = np.array([])
        self._size = size
        self._model = LinearRegression()

        self.a = np.nan
        self.b = np.nan
        self.vare = np.nan

        self.mean = np.nan
        self.sd = np.nan

    #add data
    def add(self, value):
        if len(self._observe) == 0:
            self._observe = np.array([[value]])
        else:
            self._observe = np.concatenate((self._observe, [[value]]))

        if len(self._observe) > self._size:
            self._observe = self._observe[(len(self._observe) - self._size):]

        #retrain the model
        if len(self._observe) > 3:
            x = self._observe[0:-1]
            y = self._observe[1:]
            self._model.fit(x,y)
            self.a = self._model.intercept_
            self.b = self._model.coef_
            self.vare = np.mean((self._model.predict(x) - y) ** 2)

        #get params
        if abs(self.b) < 1.0:
            self.mean = self.a/(1-self.b)
            self.sd = sqrt(self.vare/(1 - self.b * self.b))
        else:
            self.mean = np.nan
            self.sd = np.nan

    #compare
    def test_sigma(self, res, bollinger):
        if abs(self.b) < 1.0:
            return res > self.mean + bollinger * self.sd
        else:
            return False

    #get length
    def get_length(self):
        return len(self._observe)

## pair trading with OU Model
# Max position within 1

class OUAlgo(PairAlgoWrapper):

    # called when algo param is set
    def param_updated(self):
        # make sure parent updates its param
        super(OUAlgo, self).param_updated()

        # create autoregressive
        self.long_autoreg = AutoregOU(size = self.param['rolling'])
        self.short_autoreg = AutoregOU(size = self.param['rolling'])

        # create rolling
        self.spreadx_roll = SimpleMoving(size = self.param['rolling'])
        self.spready_roll = SimpleMoving(size = self.param['rolling'])

        #params
        self.bollinger = self.param['bollinger']
        self.block = self.param['block']

        #other params
        self.last_long_res = -999
        self.last_short_res = -999

        #records
        self.records = {'timestamp': [], 'longs': [], 'shorts': [],
                        'long_mean': [], 'short_mean': [],
                        'long_sd': [], 'short_sd':[]}

        #tracker
        self.tracker = self.param['tracker']

    def on_tick(self, multiple, contract, info):
        self.tracker.tick_pass_by()
        # skip if price_table doesnt have both, TODO fix this bug internally
        if self.price_table.get_size() < 2:
            return

        # get residuals and position
        long_res = self.pair.get_long_residual()
        short_res = self.pair.get_short_residual()
        pos = self.position_y()

        #two spread
        spreadx = self.spreadx_roll.mean
        spready = self.spready_roll.mean
        avg_spread = (spreadx + spready)/2

        #fee
        fee = self.pair.get_fee()

        #update record
        #self._update_record(float(long_res), float(self.long_autoreg.mean), float(self.long_autoreg.sd),\
        #                    float(short_res), float(self.short_autoreg.mean), float(self.short_autoreg.sd))

        #calculate profit
        profit = 0
        if pos == -1:
            profit = long_res + self.last_short_res
        elif pos == 1:
            profit = short_res + self.last_long_res


        # action only when unblocked: bock size < rolling queue size
        if self.long_autoreg.get_length() > self.block:
            # long when test long_res > mean+bollinger*sd
            if self.long_autoreg.test_sigma(long_res, self.bollinger) \
               and long_res > self.long_autoreg.mean + avg_spread + fee/2:
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)
                    self.last_long_res = long_res

                    #tell the tracker
                    if pos == 0:
                        self.tracker.open_position()
                    else:
                        self.tracker.close_with_exit(profit - fee)


            # short when test short_res > mean + bollinger*sd
            elif self.short_autoreg.test_sigma(short_res, self.bollinger) \
                 and short_res > self.short_autoreg.mean + avg_spread + fee/2:
                # only short when position is 0 or 1
                if pos >= 0:
                    self.short_y(y_qty=1)
                    self.last_short_res = short_res

                    #tell the tracker
                    if pos == 0:
                        self.tracker.open_position()
                    else:
                        self.tracker.close_with_exit(profit - fee)



        # update rolling
        self.long_autoreg.add(long_res)
        self.short_autoreg.add(short_res)

        self.spreadx_roll.add(self.pair.get_spread_x())
        self.spready_roll.add(self.pair.get_spread_y())

    def on_daystart(self, date, info_x, info_y):
        # recreate rolling at each day start
        self.long_autoreg = AutoregOU(size=self.param['rolling'])
        self.short_autoreg = AutoregOU(size=self.param['rolling'])

        self.spreadx_roll = SimpleMoving(size = self.param['rolling'])
        self.spready_roll = SimpleMoving(size = self.param['rolling'])

    def on_dayend(self, date, info_x, info_y):
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
    tracker = TradeAnalysis(Contract(pair[0]))
    algo = { 'class': OUAlgo }
    algo['param'] = {'x': pair[0],
                     'y': pair[1],
                     'a': 1,
                     'b': 0,
                     'rolling': param[0],
                     'bollinger': param[1],
                     'block': 100,
                     'tracker': tracker
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
    order_win = tracker.order_winning_ratio()
    order_profit = tracker.analyze_all_profit()[0]
    num_rounds = tracker.analyze_all_profit()[2]
    return score, order_win, order_profit, num_rounds

def run_simulation(param, date_list, product):
    pnl_list = []
    order_win_list = []
    order_profit_list = []
    num_rounds_list = []
    for date in date_list:
        date_pair = get_best_pair(date,market, product)
        if type(date_pair) != tuple:
            continue
        else:
            result = back_test(date_pair, date, param)
            pnl_list.append(result[0])
            order_win_list.append(result[1])
            order_profit_list.append(result[2])
            num_rounds_list.append(result[3])
    return pnl_list, order_win_list, order_profit_list, num_rounds_list

date_list = [str(x).split(' ')[0] for x in pd.date_range('2015-01-01','2016-03-31').tolist()]
roll_list = np.arange(500, 5500, 500)
sd_list = np.arange(0.5, 4.1, 0.25)
pars = list(itertools.product(roll_list, sd_list))
num_cores = 32
product = 'au'
trade_day_list = []
second_contract_size_list = []
for date in date_list:
    date_pair = get_best_pair(date, market, product)
    if type(date_pair) != tuple:
        continue
    else:
        trade_day_list.append(date)
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param,\
    date_list, product) for param in pars)
keys = ['roll:{}_sd:{}'.format(*p) for p in pars]
pnl = [i[0] for i in results]
order_win = [i[1] for i in results]
order_profit = [i[2] for i in results]
num_rounds = [i[3] for i in results]

pnl_dict = dict(zip(keys, pnl))
pnl_result = pd.DataFrame(pnl_dict)
pnl_result.index = trade_day_list
pnl_result.to_csv('./out/{}_OU_daily_pnl.csv'.format(product))

order_win_dict = dict(zip(keys, order_win))
order_win_result = pd.DataFrame(order_win_dict)
order_win_result.index = trade_day_list
order_win_result.to_csv('./out/{}_OU_daily_order_win.csv'.format(product))

order_profit_dict = dict(zip(keys, order_profit))
order_profit_result = pd.DataFrame(order_profit_dict)
order_profit_result.index = trade_day_list
order_profit_result.to_csv('./out/{}_OU_daily_order_profit.csv'.format(product))

num_rounds_dict = dict(zip(keys, num_rounds))
num_rounds_result = pd.DataFrame(num_rounds_dict)
num_rounds_result.index = trade_day_list
num_rounds_result.to_csv('./out/{}_OU_daily_num_rounds.csv'.format(product))
