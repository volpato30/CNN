import numpy as np
def iterate_minibatches(inputs, targets, batchsize, window):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - window - batchsize + 2, batchsize):
        batch = np.zeros((batchsize,window,80),dtype = np.float32)
        label = np.zeros(batchsize,dtype = np.int8)
        for i in range(batchsize):
            batch[i] = inputs[(start_idx+i):(start_idx+i+window),:]
            label[i] = targets[start_idx+window+i-1]
        yield batch, label
