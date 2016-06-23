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
from my_utils import get_trade_day, contract_backtest, get_price_diff
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

pair = ['if1604', 'if1605']
date_list = [str(x).split(' ')[0] for x in pd.date_range('2016-03-01','2016-03-31').tolist()]
algo = { 'class': TestAlgo }
algo['param'] = {'x': pair[0],
                 'y': pair[1],
                 'a': 1,
                 'b': 0,
                 'rolling': 4000,
                 'sd_coef': 3,
                 'block': 100,
                 }
settings = { 'date': date_list,
             'path': DATA_PATH,
             'tickset': 'top',
             'algo': algo}
runner = PairRunner(settings)
rolling_list = range(500,5000,500)
sd_coef_list = np.arange(1, 5, 0.5)
final_profit = []
for r in rolling_list :
    for sd in sd_coef_list :
        start_time = time.time()
        runner.run(algo_param={'rolling': r, 'sd_coef': sd})
        account = runner.account
        history = account.history.to_dataframe(account.items)
        score = float(history[['pnl']].iloc[-1])
        final_profit.append(score)
        print("rolling {}, sd_coef {}, backtest took {:.3f}s, score is {:.3f}".format(r, sd, time.time() - start_time, score))
pars = list(itertools.product(rolling_list, sd_coef_list))
result = pd.DataFrame({"rolling": [p[0] for p in pars],
                       "sd_coef": [p[1] for p in pars],
                       "PNL": [float(f) for f in final_profit]})
pickle.dump(result, open( "if_16Mar.p", "wb" ))
