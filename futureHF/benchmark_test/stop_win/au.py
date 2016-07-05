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
from mewp.data.item import Contract

from joblib import Parallel, delayed
import datetime
import numpy as np
import pandas as pd
import itertools
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

def get_tracker(date_list, product):
    pair = 0
    for date in date_list:
        pair = get_best_pair(date,market, product)
        if type(pair) != tuple:
            continue
        else:
            break
    return TradeAnalysis(Contract(pair[0]))

#algo
## pair trading with stop win and SMA
# Max position within 1
class StopWinSpreadGuardAlgo(PairAlgoWrapper):

    # called when algo param is set
    def param_updated(self):
        # make sure parent updates its param
        super(StopWinSpreadGuardAlgo, self).param_updated()

        # algo settings

        self.min_ticksize = self.pair.x.symbol.min_ticksize
        # create rolling
        self.long_roll = SimpleMoving(size=self.param['rolling'])
        self.short_roll = SimpleMoving(size=self.param['rolling'])

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
        self.tracker = self.param['tracker']

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
        trade_flag = 0

        ## only do this when plotting is needed
        #update record
#       self._update_record(long_res, self.long_roll.mean, self.long_roll.sd,\
#                           short_res, self.short_roll.mean, self.short_roll.sd)

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

        #stop win
        # stop short position
        if pos == -1:
            if profit >= max(self.min_ticksize, self.stop_win * self.long_roll.sd, fee):
                self.long_y(y_qty = 1)
                self.last_long_res = long_res
                self.tracker.close_with_stop(profit - fee)
                trade_flag = 1

        # stop long position
        if pos == 1:
            if profit >= max(self.min_ticksize, self.stop_win * self.short_roll.sd, fee):
                self.short_y(y_qty = 1)
                self.last_short_res = short_res
                self.tracker.close_with_stop(profit - fee)
                trade_flag = 1

        # open or close position
        # action only when unblocked: bock size < rolling queue size
        if self.long_roll.queue.qsize() > self.block and trade_flag == 0:
            # long when test long_res > mean+bollinger*sd
            if self.long_roll.test_sigma(long_res, self.bollinger) \
                   and long_res - self.long_roll.mean > fee + avg_spread:
                # only long when position is 0 or -1
                if pos <= 0:
                    self.long_y(y_qty=1)
                    self.last_long_res = long_res

                    #tell the tracker
                    if pos == 0:
                        self.tracker.open_position()
                    else:
                        self.tracker.close_with_exit(profit - fee)

            # short when test short_res > mean+bollinger*sd
            elif self.short_roll.test_sigma(short_res, self.bollinger) \
                     and short_res - self.short_roll.mean > fee + avg_spread:
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

def back_test(pair, date, param, tracker):
    algo = { 'class': StopWinSpreadGuardAlgo }
    algo['param'] = {'x': pair[0],
                     'y': pair[1],
                     'a': 1,
                     'b': 0,
                     'rolling': param[0],
                     'bollinger': param[1],
                     'stop_win': param[2],
                     'block': 100,
                     'tracker': tracker}
    settings = { 'date': date,
                 'path': DATA_PATH,
                 'tickset': 'top',
                 'algo': algo,
                 'singletick': True}
    runner = PairRunner(settings)
    runner.run()
    return runner, algo
#
def run_simulation(param, date_list, product):
    order_win_list = []
    daily_num_order = []
    order_waiting_list = []
    order_profit_list = []
    master = MasterReport()
    tracker = get_tracker(date_list, product)


    for date in date_list:
        date_pair = get_best_pair(date, market, product)
        if type(date_pair) != tuple:
            continue
        else:
            runner, _ = back_test(date_pair, date, param, tracker)
            try:
                report = Report(runner)
            except IndexError:
                print 'WTF? {} has IndexError'.format(date)
                continue
            report.print_report(to_file=False, to_screen=False, to_master=master)

    try:
        [overall, days] = master.print_report(to_file=False, print_days=False)
    except TypeError as inst:
        if inst.args[0] == "'NoneType' object has no attribute '__getitem__'":
            return ('NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA')
        else:
            raise Exception("god knows what happens")

    #pnls
    final_pnl = float(overall.final_pnl)
    final_return = float(overall.final_return)
    sharpe_ratio = float(overall.sharpe_ratio)
    win_ratio = float(overall.win_ratio)

    #max draw down
    daily_draw_down = np.asarray(days.max_draw_down)
    max_draw_down = daily_draw_down.max()
    avg_draw_down = daily_draw_down.mean()

    #num orders
    num_orders = sum(days.order_count)

    #order analysis
    order_win = tracker.order_winning_ratio()
    order_waiting = tracker.analyze_all_waiting()[0]
    order_waiting_median = tracker.analyze_all_waiting()[3]
    order_profit = tracker.analyze_all_profit()[0]
    order_profit_median = tracker.analyze_all_profit()[3]
    num_rounds = tracker.analyze_all_profit()[2]

    return final_pnl, final_return, sharpe_ratio, win_ratio, max_draw_down,\
        avg_draw_down, num_orders, num_rounds, order_win, order_waiting, order_waiting_median, \
        order_profit, order_profit_median


date_list = [str(x).split(' ')[0] for x in pd.date_range('2016-01-01','2016-03-31').tolist()]
roll_list = np.arange(500, 11000, 500)
sd_list = np.arange(0.5, 4.1, 0.25)
stop_win_list = np.arange(0.5, 4.1, 0.25)
num_cores = 20
product = 'au'
pars = list(itertools.product(roll_list, sd_list, stop_win_list))
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param,\
    date_list, product) for param in pars)
result = pd.DataFrame({ "aaa_rolling": [p[0] for p in pars],
                        "aaa_bollinger": [p[1] for p in pars],
                        "aaa_stop_win": [p[2] for p in pars],
                        "PNL": [i[0] for i in results],
                        "return": [i[1] for i in results],
                        "sharpe_ratio": [i[2] for i in results],
                        "win_ratio": [i[3] for i in results],
                        "max_draw_down": [i[4] for i in results],
                        "average_draw_down": [i[5] for i in results],
                        "num_orders": [i[6] for i in results],
                        "num_rounds": [i[7] for i in results],
                        "order_win": [i[8] for i in results],
                        "order_waiting": [i[9] for i in results],
                        "order_waiting_median": [i[10] for i in results],
                        "order_profit": [i[11] for i in results],
                        "order_profit_median": [i[12] for i in results]})
result.to_csv('./out/' + '{}_SG_SW'.format(product) + '.csv')
