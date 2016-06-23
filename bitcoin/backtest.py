def backtest(merged, signal,start_cash):
    PNL = 100000 * np.ones(len(merged))
    position = np.zeros((len(merged),2))#column 0 is position_x, column 1 is position_y
    cash = 100000 * np.ones(len(merged))
    num_trade = 0
    count = np.round(len(merged) / 10)
    for i in range(1,len(merged)):
        if i % count == count-1:
            print "{}%".format((i+1)/count*10)
        id_x = merged.ix[merged.index[i],2]
        ask_x = merged.ix[merged.index[i],3]
        bid_y = merged.ix[merged.index[i],6]
        ask_y = merged.ix[merged.index[i],7]
        if signal[i] == 0:
            position[i,0] = position[i-1,0]
            position[i,1] = position[i-1,1]
            cash[i] = cash[i-1]
            PNL[i] = pnl(cash[i],position[i,0],position[i,1], bid_x, ask_x, bid_y, ask_y)
        elif signal[i] == 1:
            if position[i-1,0] == 1:
                position[i,0] = position[i-1,0]
                position[i,1] = position[i-1,1]
                cash[i] = cash[i-1]
                PNL[i] = pnl(cash[i],position[i,0],position[i,1], bid_x, ask_x, bid_y, ask_y)
            else:
                num_trade += 1
                position[i,0] = 1
                position[i,1] = -1
                cash[i] = cash[i-1] - (1-position[i-1,0]) * ask_x + (position[i-1,1]+1) * bid_y
                PNL[i] = pnl(cash[i],position[i,0],position[i,1], bid_x, ask_x, bid_y, ask_y)
        else:
            if position[i-1,1] == 1:
                position[i,0] = position[i-1,0]
                position[i,1] = position[i-1,1]
                cash[i] = cash[i-1]
                PNL[i] = pnl(cash[i],position[i,0],position[i,1], bid_x, ask_x, bid_y, ask_y)
            else:
                num_trade += 1
                position[i,0] = -1
                position[i,1] = 1
                cash[i] = cash[i-1] + (position[i-1,0] + 1) * bid_x + (1 - position[i-1,1]) * ask_y
                PNL[i] = pnl(cash[i],position[i,0],position[i,1], bid_x, ask_x, bid_y, ask_y)
    return PNL
