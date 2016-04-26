
# coding: utf-8

# In[1]:

import sqlite3
import numpy as np
import pandas as pd


# In[2]:

def db_readDbConfig(connection):
    c = connection.cursor()
    c.execute("SELECT cast(Timestamp as float), * FROM config")
    config = c.fetchall()
    return config[0]


# In[5]:

def db_readDepthHour(filename):
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    dbconfig = db_readDbConfig(conn)
    timeOffset = dbconfig[0]
    depthSize = dbconfig[4]
    priceScale = dbconfig[5]
    amountScale = dbconfig[6]
    depthSizes = np.arange(depthSize)
    c.execute("SELECT * FROM Data")
    data = c.fetchall()
    data = np.asarray(data)
    Orderbook = data[:,1:].astype(np.float32)
    priceIndex = np.linspace(0,78,40,dtype=np.int8)
    amountIndex = np.linspace(1,79,40,dtype=np.int8)
    timestamp = data[:, 0] + timeOffset
    Orderbook[:,priceIndex] = Orderbook[:, priceIndex] / np.asarray(priceScale,dtype = np.float32)
    Orderbook[:,amountIndex] = Orderbook[:, amountIndex] / np.asarray(amountScale,dtype = np.float32)
    conn.close()
    return Orderbook, timestamp

