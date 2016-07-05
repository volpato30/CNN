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
from my_utils import OUAlgo
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
roll_list = np.arange(500, 8500, 500)
sd_list = np.arange(0.5, 4.1, 0.25)
pars = list(itertools.product(roll_list, sd_list))
num_cores = 32
product = 'ni'
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
