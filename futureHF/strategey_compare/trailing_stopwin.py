from sys import path
path.append('/work/rqiao/HFdata')
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.runner import PairRunner
from mewp.math.simple import SimpleMoving
from mewp.util.clock import Clock
from mewp.data.order import OrderType
from mewp.simulate.report import Report
from mewp.util.pair_trade_analysis import TradeAnalysis
from joblib import Parallel, delayed
from math import sqrt
import datetime
import pandas as pd
import numpy as np
import itertools
import pickle
DATA_PATH = '/work/rqiao/HFdata/dockfuture'

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
        self.spreadx_roll = SimpleMoving(size = self.param['rolling'])
        self.spready_roll = SimpleMoving(size = self.param['rolling'])

        #params
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

        self.max_profit = 0

        #tracker
        self.tracker = TradeAnalysis(self.pair.x)

    def on_tick(self, multiple, contract, info):
        self.tracker.tick_pass_by()
        # skip if price_table doesnt have both, TODO fix this bug internally
        if len(self.price_table.table) < 2:
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
#         self._update_record(long_res, self.long_roll.mean, self.long_roll.sd,\
#                             short_res, self.short_roll.mean, self.short_roll.sd)

        #calculate profit
        profit = 0
        if pos == -1:
            profit = long_res + self.last_short_res
        elif pos == 1:
            profit = short_res + self.last_long_res

        #trailing stop win
        if profit > self.max_profit and profit > 0:
            self.max_profit = profit
        else:
            # stop short position
            if pos == -1:
                if self.max_profit - profit > max(1,self.stop_win * self.long_roll.sd) and profit > 0:
                    self.long_y(y_qty = 1)
                    self.last_long_res = long_res
                    self.tracker.close_with_stop(profit - fee)
                    return

            # stop long position
            if pos == 1:
                if self.max_profit - profit > max(1,self.stop_win * self.short_roll.sd) and profit > 0:
                    self.short_y(y_qty = 1)
                    self.last_short_res = short_res
                    self.tracker.close_with_stop(profit - fee)
                    return

        # action only when unblocked: bock size < rolling queue size
        if self.long_roll.queue.qsize() > self.block:
            # long when test long_res > mean+bollinger*sd
            if self.long_roll.test_sigma(long_res, self.bollinger) \
               and long_res > self.long_roll.mean + avg_spread + fee/2:
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)
                    self.last_long_res = long_res

                    self.max_profit = 0

                    #tell the tracker
                    if pos == 0:
                        self.tracker.open_position()
                    else:
                        self.tracker.close_with_exit(profit - fee)

                    return

            # short when test short_res > mean+bollinger*sd
            elif self.short_roll.test_sigma(short_res, self.bollinger) \
                 and short_res > self.short_roll.mean + avg_spread + fee/2:
                # only short when position is 0 or 1
                if pos >= 0:
                    self.short_y(y_qty=1)
                    self.last_short_res = short_res

                    self.max_profit = 0

                    #tell the tracker
                    if pos == 0:
                        self.tracker.open_position()
                    else:
                        self.tracker.close_with_exit(profit - fee)

                    return
            else:
                pass


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

algo = { 'class': SMAAlgo }
algo['param'] = {'x': 'a1505',
                 'y': 'a1509',
                 'a': 1,
                 'b': 0,
                 'rolling': 1000,
                 'bollinger': 2,
                 'block': 100,
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
bollinger_list = range(2,8,1)
stop_win_list = range(1,9,1)
final_profit = []
num_trades = []
pars = list(itertools.product(rolling_list, bollinger_list,stop_win_list))

def run_simulation(p):
    runner.run(algo_param={'rolling': p[0], 'bollinger': p[1], 'stop_win': p[2]})
    report = Report(runner)
    runner._algo.tracker.order_winning_ratio()
    return report.get_final_pnl(), report.get_final_return(), report.get_sharpie_ratio(),\
        report.get_avg_max_draw_down(), report.get_max_max_draw_down()[0], \
        runner._algo.tracker.order_winning_ratio(), runner._algo.tracker.analyze_all_waiting()[0],\
        runner._algo.tracker.analyze_all_profit()[0], runner._algo.tracker.analyze_all_profit()[2]
num_cores = 32
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(p) for p in pars)
result = pd.DataFrame({"rolling": [p[0] for p in pars],
                       "bollinger": [p[1] for p in pars],
                       "stop_win": [p[2] for p in pars],
                       "PNL": [i[0] for i in results],
                       "return": [i[1] for i in results],
                       "sharpe_ratio": [i[2] for i in results],
                       "avg_draw_down": [i[3] for i in results],
                       "max_draw_down": [i[4] for i in results],
                       "order_winning_ratio": [i[5] for i in results],
                       "avg_waiting": [i[6] for i in results],
                       "avg_profit": [i[7] for i in results],
                       "num_trades": [i[8] for i in results]})
result.to_csv('trailing_stopwin_result.csv')
