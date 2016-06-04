from sys import path
path.append('/work/rqiao/HFdata')
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.runner import PairRunner
from mewp.math.simple import SimpleMoving
from mewp.util.clock import Clock
from mewp.data.order import OrderType
import time
import matplotlib
import matplotlib.pylab as plt
import itertools
import pickle
import sys
import os
import numpy as np
import pandas as pd
import datetime
from numpy import cumsum, log, sqrt, std, subtract
from Queue import Queue
from mewp.util.futures import get_day_db_path
from mewp.reader.futuresqlite import SqliteReaderDce, SqliteReaderL1
from mewp.data.frame import *
from sqlite3 import OperationalError
from my_utils import get_best_pair
from joblib import Parallel, delayed
import itertools
DATA_PATH = '/work/rqiao/HFdata/dockfuture'

class TestAlgo(PairAlgoWrapper):

    # called when algo param is set
    def param_updated(self):
        # make sure parent updates its param
        super(TestAlgo, self).param_updated()

        # create rolling
        self.sd_coef = self.param['sd_coef']
        self.block = self.param['block']

    def on_daystart(self, date, info_x, info_y):
        # recreate rolling at each day start
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])

    def on_dayend(self, date, info_x, info_y):
        pass

    def on_tick(self, multiple, contract, info):
        # skip if hasn't receive price on both side
        if len(self.price_table.table) < 2:
            return

        # get residuals and position
        long_res = self.pair.get_long_residual()
        short_res = self.pair.get_short_residual()
        pos = self.position_y()
        # action only when unblocked: bock size < rolling queue size
        if self.long_roll.queue.qsize() > self.block:
            # long when test long_res > roll.mean+sd_coef*roll.sd
            if self.long_roll.test_sigma(long_res, self.sd_coef):
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)

            # short when test short_res > roll.mean+sd_coef*roll.sd
            elif self.short_roll.test_sigma(short_res, self.sd_coef):
                 # only short when position is 0 or 1
                if pos >= 0:
                    self.short_y(y_qty=1)
            else:
                pass
        # update rolling
        self.long_roll.add(long_res)
        self.short_roll.add(short_res)


def back_test(pair, date, param):
    algo = { 'class': TestAlgo }
    algo['param'] = {'x': pair[0],
                     'y': pair[1],
                     'a': 1,
                     'b': 0,
                     'rolling': param[0],
                     'sd_coef': param[1],
                     'block': 100,
                     }
    settings = { 'date': date,
                 'path': DATA_PATH,
                 'tickset': 'top',
                 'algo': algo}
    runner = PairRunner(settings)
    runner.run()
    account = runner.account
    history = account.history.to_dataframe(account.items)
    orders = account.orders.to_dataframe()
    pnl = np.asarray(history.pnl)[-1]
    return pnl, len(orders)

def run_simulation(param, date_list):
    total_pnl = 0
    total_order = 0
    for date in date_list:
        date_pair = get_best_pair(date,'if')
        if type(date_pair) != tuple:
            continue
        else:
            tpnl, torder = back_test(date_pair, date, param)
            total_pnl += tpnl
            total_order += torder
    return total_pnl, total_order

roll_list = np.arange(500, 4100, 500)
sd_list = np.arange(1, 4.1, 0.2)
pars = list(itertools.product(roll_list, sd_list))
date_list = [str(x).split(' ')[0] for x in pd.date_range('2016-01-01','2016-03-31').tolist()]
num_cores = 32
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param, date_list) for param in pars)
result = pd.DataFrame({"rolling": [p[0] for p in pars],
                       "sd_coef": [p[1] for p in pars],
                       "PNL": [i for i, v in results],
                       "num_trades": [v for i, v in results]})
pickle.dump(result, open('if_result.p','wb'))
