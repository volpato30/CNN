import pylab
import pickle
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
from sklearn import datasets
# load the saved model
layer0,layer1,layer2,layer3 = pickle.load(open('weight.pkl','rb'))

face=datasets.fetch_olivetti_faces(shuffle=True)
x=face.data[0,:]
x=x.reshape(1,1,64,64)

input = T.tensor4(name='input')

pooled_out = downsample.max_pool_2d(
            input=input,
            ds=(64,64),
            ignore_border=True,
            mode='average_exc_pad'
        )
f = theano.function([input], pooled_out)
print(f(x))
