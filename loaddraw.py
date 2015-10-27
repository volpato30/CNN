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

conv_out = conv.conv2d(input,filters=layer0.params[0])
pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=(2,2),
            ignore_border=True
        )
output = T.tanh(pooled_out + layer0.params[1].dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)
filtered_img = f(x)
pylab.gray();
pylab.subplot(1, 3, 1) 
pylab.axis('off')
pylab.imshow(x[0, 0, :, :])
pylab.subplot(1, 3, 2) 
pylab.axis('off')
pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3) 
pylab.axis('off')
pylab.imshow(filtered_img[0, 1, :, :])

pylab.show()