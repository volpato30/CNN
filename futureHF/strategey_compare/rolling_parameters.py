from sys import path
path.append('/work/rqiao/HFdata')
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.runner import PairRunner
from mewp.math.simple import SimpleMoving
from joblib import Parallel, delayed
import sys
import os
import itertools
import numpy as np
import pandas as pd
import datetime as dt
from mewp.util.futures import get_day_db_path
from mewp.reader.futuresqlite import SqliteReaderDce, SqliteReaderL1
from sqlite3 import OperationalError
from my_utils import get_trade_day
import pickle
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

pair = ['a1505', 'a1509']
sd_coef = np.arange(2.0, 6.1, 0.5)
rolling = np.arange(1000, 9000, 1000)
Date_list = get_trade_day(pair)
pars = list(itertools.product(rolling, sd_coef))

def run_simulation(p,runner):
    runner.run(algo_param={'alpha': p[0], 'stop_win': p[1]})
    account = runner.account
    orders = account.orders.to_dataframe()
    history = account.history.to_dataframe(account.items)
    return float(history.pnl.tail(1)), len(orders)

def best_param(date_list):
    score = []
    num_trades = []
    algo = { 'class': TestAlgo }
    algo['param'] = {'x': pair[0],
                    'y': pair[1],
                    'a': 1,
                    'b': 0,
                    'rolling': 2000,
                    'sd_coef': 2,
                    'block': 100}
    settings = { 'date': date_list,
                'path': DATA_PATH,
                'tickset': 'top',
                'algo': algo}
    runner = PairRunner(settings)
    num_cores = 32
    results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(p, runner) for p in pars)
    result = pd.DataFrame({ "rolling": [p[0] for p in pars],
                            "sd_coef": [p[1] for p in pars],
                            "PNL": [i for i, v in results],
                            "num_trades": [v for i, v in results]})
    index = np.asarray(result.PNL).argmax()
    return result[index:index+1]

def run_based_on_pre(date_list):
    temp = best_param(date_list[:-1])
    algo = { 'class': TestAlgo }
    algo['param'] = {'x': pair[0],
                    'y': pair[1],
                    'a': 1,
                    'b': 0,
                    'rolling': float(temp.rolling),
                    'sd_coef': float(temp.sd_coef),
                    'block': 100}
    settings = { 'date': date_list[-1],
                'path': DATA_PATH,
                'tickset': 'top',
                'algo': algo}
    runner = PairRunner(settings)
    runner.run()
    account = runner.account
    orders = account.orders.to_dataframe()
    history = account.history.to_dataframe(account.items)
    return float(history.pnl.tail(1)), len(orders)

def simul_run(pre_days = 3):
    temp = best_param(Date_list[:pre_days])
    total_pnl = float(temp.PNL)
    total_num_trades = float(temp.num_trades)
    for i in range(len(Date_list)-pre_days):
        temp_pnl, temp_num_orders = run_based_on_pre(Date_list[i:i+pre_days+1])
        total_pnl += temp_pnl
        total_num_trades += temp_num_orders
    print 'use parameters from previous {} days, total pnl : {}, \
        total num_trades: {}'.format(pre_days, total_pnl, total_num_trades)

simul_run(1)
