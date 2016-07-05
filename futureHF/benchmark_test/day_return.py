from sys import path
path.append('/work/rqiao/HFdata/cython_mew-p')
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
from my_utils import SpreadGuardAlgo
from joblib import Parallel, delayed
import datetime
import numpy as np
import pandas as pd
import itertools
import os
DATA_PATH = '/work/rqiao/HFdata/dockfuture'
market = 'shfe'

def get_file_size(market, contract, date):
    statinfo = os.stat(get_day_db_path(DATA_PATH, contract, date))
    return statinfo.st_size

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
#
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
roll_list = np.concatenate((np.arange(200,500,100), np.arange(500, 4500, 500)))
sd_list = np.concatenate((np.arange(0.2,0.5,0.1), np.arange(0.5, 3.1, 0.25)))
product_list = ['al','ag','au','cu','pb','ru','zn']
pars = list(itertools.product(roll_list, sd_list))
num_cores = 20
for product in product_list:
#get trade_day_list
    trade_day_list = []
    second_contract_size_list = []
    for date in date_list:
        date_pair = get_best_pair(date, market, product)
        if type(date_pair) != tuple:
            continue
        else:
            trade_day_list.append(date)
            second_contract_size_list.append(get_file_size(market, date_pair[1], date))
    results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param,\
        date_list, product) for param in pars)
    keys = ['roll:{}_sd:{}'.format(*p) for p in pars]
    dictionary = dict(zip(keys, results))
    result = pd.DataFrame(dictionary)
    result['second_contract_size'] = second_contract_size_list
    result.index = trade_day_list
    result.to_csv('{}_day_return.csv'.format(product))
