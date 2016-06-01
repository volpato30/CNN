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

class Moving(object):
    ##constructor
    # @param size of moving window
    def __init__(self, size, sigma_size):
        self.size = size
        self.sigma_size = sigma_size
        self.queue = Queue()
        self.sigma_queue = Queue()
        self.sums = 0
        self.mean = 0
        self.powersum = 0
        self.var = 0
        self.std = 0

    ## add a new observation
    # @param observe new observation
    # @param mean mean at that moment
    def add(self,observe):
        self.sums += observe
        self.powersum += observe**2
        self.queue.put(observe)
        self.sigma_queue.put(observe)
        while self.queue.qsize() > self.size:
            popped = self.queue.get()
            self.sums -= popped
        self.mean = self.sums / self.queue.qsize()
        while self.sigma_queue.qsize() > self.sigma_size:
            popped = self.sigma_queue.get()
            self.powersum -= popped ** 2
        size = self.sigma_queue.qsize()
        self.var = (self.powersum - size*self.mean*self.mean) / size
        self.std = sqrt(self.var)

    # return the standard deviation
    # @param mean mean for this moment
    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean

class TestAlgo(PairAlgoWrapper):

    # called when algo param is set
    def param_updated(self):
        # make sure parent updates its param
        super(MyAlgo, self).param_updated()
        # create rolling
        self.sd_coef = self.param['sd_coef']
        self.block = self.param['block']

    def on_daystart(self, date, info_x, info_y):
        # recreate rolling at each day start
        self.long_roll = Moving(size = self.param['rolling'], sigma_size = self.param['rolling_sigma'])
        self.short_roll = Moving(size = self.param['rolling'], sigma_size = self.param['rolling_sigma'])

    def on_tick(self, multiple, contract, info):
        # skip if price_table doesnt have both, TODO fix this bug internally
        if len(self.price_table.table) < 2:
            return

        # get residuals and position
        long_res = self.pair.get_long_residual()
        short_res = self.pair.get_short_residual()
        pos = self.position_y()

        # update rolling
        self.long_roll.add(long_res)
        self.short_roll.add(short_res)
        long_mean = self.long_roll.get_mean()
        short_mean = self.short_roll.get_mean()
        long_std = self.long_roll.get_std()
        short_std = self.short_roll.get_std()

        # action only when unblocked: bock size < rolling queue size
        if self.long_roll.queue.qsize() > self.block:
            # long when test long_res > autoreg.mean+sd_coef*roll.sd
            if long_res > long_mean + self.sd_coef * long_std:
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)
            # short when test short_res > autoreg.mean+sd_coef*roll.sd
            elif short_res > short_mean + self.sd_coef * short_std:
                 # only short when position is 0 or 1
                if pos >= 0:
                    self.short_y(y_qty=1)
            else:
                pass
    def on_dayend(self, date, info_x, info_y):
        pass

pair = ['if1602', 'if1603']
date_list = [str(x).split(' ')[0] for x in pd.date_range('2016-01-01','2016-02-10').tolist()]
algo = { 'class': TestAlgo }
algo['param'] = {'x': pair[0],
                 'y': pair[1],
                 'a': 1,
                 'b': 0,
                 'rolling': 1000,
                 'rolling_sigma': 1000,
                 'sd_coef': 3,
                 'block': 100,
                 }
settings = { 'date': date_list,
             'path': DATA_PATH,
             'tickset': 'top',
             'algo': algo}
runner = PairRunner(settings)
rolling_list = range(500,2500,500)
rolling_sigma_list = range(100,1000,100)
sd_coef_list = np.arange(1.5, 3, 0.1)
final_profit = []
for r in rolling_list :
    for rs in rolling_sigma_list:
        for sd in sd_coef_list :
            start_time = time.time()
            runner.run(algo_param={'rolling': r,'rolling_sigma': rs, 'sd_coef': sd})
            account = runner.account
            history = account.history.to_dataframe(account.items)
            score = float(history[['pnl']].iloc[-1])
            final_profit.append(score)
            print("rolling {}, rolling sigma {}, sd_coef {}, backtest took {:.3f}s, score is {:.3f}".format(r, rs, sd, time.time() - start_time, score))
pars = list(itertools.product(rolling_list, rolling_sigma_list, sd_coef_list))
result = pd.DataFrame({"rolling": [p[0] for p in pars],
                       "rolling": [p[1] for p in pars],
                       "sd_coef": [p[2] for p in pars],
                       "PNL": [float(f) for f in final_profit]})
pickle.dump(result, open( "if_16Jan.p", "wb" ))
