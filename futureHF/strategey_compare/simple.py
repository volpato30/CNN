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

algo = { 'class': TestAlgo }
algo['param'] = {'x': 'a1505',
                 'y': 'a1509',
                 'a': 1,
                 'b': 0,
                 'rolling': 1000,
                 'sd_coef': 2,
                 'block': 100,
                 }

dates = pd.date_range('20150101', '20150228')
date_list = [day.strftime("%Y-%m-%d") for day in dates]
settings = { 'date': date_list,
             'path': DATA_PATH,
             'tickset': 'top',
             'algo': algo}
runner = PairRunner(settings)
rolling_list = range(1000,9000,1000)
sd_coef_list = range(2,15,1)
final_profit = []
num_trades = []
pars = list(itertools.product(rolling_list, sd_coef_list))
def run_simulation(p):
    runner.run(algo_param={'rolling': p[0], 'sd_coef': p[1]})
    account = runner.account
    orders = account.orders.to_dataframe()
    history = account.history.to_dataframe(account.items)
    return float(history.pnl.tail(1)), len(orders)
num_cores = 32
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(p) for p in pars)
result = pd.DataFrame({"rolling": [p[0] for p in pars],
                       "sd_coef": [p[1] for p in pars],
                       "PNL": [i for i, v in results],
                       "num_trades": [v for i, v in results]})
pickle.dump(result, open('Grid_search_a.p','wb'))
