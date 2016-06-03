from sys import path
path.append('/work/rqiao/HFdata')
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.runner import PairRunner
from mewp.math.simple import SimpleMoving
from mewp.util.clock import Clock
from mewp.data.order import OrderType
import time
import itertools
import pickle
import os
import numpy as np
import pandas as pd
import datetime
from Queue import Queue
from mewp.util.futures import get_day_db_path
from mewp.reader.futuresqlite import SqliteReaderDce, SqliteReaderL1
from mewp.data.frame import *
from sqlite3 import OperationalError
from my_utils import get_best_pair
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

def back_test(pair, date):
    algo = { 'class': TestAlgo }
    algo['param'] = {'x': pair[0],
                     'y': pair[1],
                     'a': 1,
                     'b': 0,
                     'rolling': 1000,
                     'sd_coef': 2,
                     'block': 100,
                     }
    settings = { 'date': date,
                 'path': DATA_PATH,
                 'tickset': 'top',
                 'algo': algo}
    runner = PairRunner(settings)
    runner.run()
    account = runner.account
    return account

date_list = [str(x).split(' ')[0] for x in pd.date_range('2015-01-01','2016-03-01').tolist()]
result = {}
account_history = {}
pnl = np.zeros(1)
for date in date_list:
    date_pair = get_best_pair(date)
    if type(date_pair) != tuple:
        continue
    else:
        pair = date_pair
        pre = pair
        account = back_test(pair,date)
        history = account.history.to_dataframe(account.items)
        temp = np.asarray(history.pnl)
        pnl = np.concatenate([pnl,pnl[-1] + temp])
        account_history[date] = account
        result[date] = temp[-1]
print 'final result'
print pnl[-1]
pickle.dump(pnl, open( "/work/rqiao/backtest_result/pnl.p", "wb" ))
pickle.dump(result, open( "/work/rqiao/backtest_result/result.p", "wb" ))
pickle.dump(account_history, open( "/work/rqiao/backtest_result/account_history.p", "wb" ))
