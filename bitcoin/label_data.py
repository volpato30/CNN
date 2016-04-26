import numpy as np
def label_data(data, timestamp, timestep, margin):
    dif = np.diff(timestamp)
    length = len(timestamp)
    label = np.zeros(length,dtype = np.int8)
    label += 9
    for l_i in range(length):
        cumsum = 0
        flag = 0
        for i in range(l_i,length-1):
            cumsum += dif[i]
            if cumsum >= timestep:
                flag = 1
                break
        if flag == 0:
            break
        if data[i+1,1] > data[l_i,3] + margin:
            label[l_i] = 0
        elif data[i+1,1] > data[l_i,3] - margin:
            label[l_i] = 1
        else:
            label[l_i] = 2
    label = label[label!=9]
    return label
