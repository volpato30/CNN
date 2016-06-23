from sys import path
path.append('/work/rqiao/HFdata')
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.runner import PairRunner
from mewp.math.simple import SimpleMoving
from mewp.util.clock import Clock
from mewp.data.order import OrderType
from joblib import Parallel, delayed
from math import sqrt
import datetime
import pandas as pd
import numpy as np
import itertools
import pickle
DATA_PATH = '/work/rqiao/HFdata/dockfuture'

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


    ## return mean
    def getMean(self):
        return self.mean
## pair trading with stop win and SMA
# Max position within 1
class SMAAlgo(PairAlgoWrapper):

    # called when algo param is set
    def param_updated(self):
        # make sure parent updates its param
        super(SMAAlgo, self).param_updated()

        # create rolling
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])
        self.bollinger = self.param['bollinger']
        self.block = self.param['block']
        self.stop_win = self.param['stop_win']

        #other params
        self.last_long_res = -999
        self.last_short_res = -999



    def on_tick(self, multiple, contract, info):
        # skip if price_table doesnt have both, TODO fix this bug internally
        if len(self.price_table.table) < 2:
            return

        # get residuals and position
        long_res = self.pair.get_long_residual()
        short_res = self.pair.get_short_residual()
        pos = self.position_y()

        # stop short position
        if pos == -1:
            if long_res + self.last_short_res >= self.stop_win * self.long_roll.sd:
                self.long_y(y_qty = 1)
                return

        # stop long position
        if pos == 1:
            if short_res + self.last_long_res >= self.stop_win * self.short_roll.sd:
                self.short_y(y_qty = 1)
                return

        # action only when unblocked: bock size < rolling queue size
        if self.long_roll.queue.qsize() > self.block:
            # long when test long_res > mean+bollinger*sd
            if self.long_roll.test_sigma(long_res, self.bollinger):
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)
                    self.last_long_res = long_res
                    return

            # short when test short_res > mean+bollinger*sd
            elif self.short_roll.test_sigma(short_res, self.bollinger):
                 # only short when position is 0 or 1
                if pos >= 0:
                    self.short_y(y_qty=1)
                    self.last_short_res = short_res
                    return
            else:
                pass


        # update rolling
        self.long_roll.add(long_res)
        self.short_roll.add(short_res)

    def on_daystart(self, date, info_x, info_y):
        # recreate rolling at each day start
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])

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

algo = { 'class': SMAAlgo }
algo['param'] = {'x': 'a1505',
                 'y': 'a1509',
                 'a': 1,
                 'b': 0,
                 'rolling': 1000,
                 'bollinger': 2,
                 'block': 100,
                 'alpha': 0.0007,
                 'stop_win': 1,
                 }

dates = pd.date_range('20150101', '20150228')
date_list = [day.strftime("%Y-%m-%d") for day in dates]

settings = { 'date': date_list,
             'path': DATA_PATH,
             'tickset': 'top',
             'algo': algo}
runner = PairRunner(settings)
rolling_list = range(1000,9000,1000)
bollinger_list = range(2,15,1)
stop_win_list = range(1,15,1)
final_profit = []
num_trades = []
pars = list(itertools.product(rolling_list, bollinger_list,stop_win_list))

def run_simulation(p):
    runner.run(algo_param={'rolling': p[0], 'bollinger': p[1], 'stop_win': p[2]})
    account = runner.account
    orders = account.orders.to_dataframe()
    history = account.history.to_dataframe(account.items)
    return float(history.pnl.tail(1)), len(orders)
num_cores = 32
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(p) for p in pars)
result = pd.DataFrame({"rolling": [p[0] for p in pars],
                       "bollinger": [p[1] for p in pars],
                       'stop_win': [p[2] for p in pars],
                       "PNL": [i for i, v in results],
                       "num_trades": [v for i, v in results]})
pickle.dump(result, open('stopwin_sma_result.p','wb'))
