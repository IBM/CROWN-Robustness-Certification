##
## Copyright (C) IBM Corp, 2018
## Copyright (C) Huan Zhang <huan@huan-zhang.com>, 2018
## Copyright (C) Tsui-Wei Weng  <twweng@mit.edu>, 2018
## 
## This program is licenced under the Apache-2.0 licence,
## contained in the LICENCE file in this directory.
##


import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import SGD
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import argparse
import os

def train(data, file_name, params, num_epochs=50, batch_size=256, train_temp=1, init=None, lr=0.01, decay=1e-5, momentum=0.9, activation="relu"):
    """
    Train a n-layer simple network for MNIST and CIFAR
    """
    
    # create a Keras sequential model
    model = Sequential()
    # reshape the input (28*28*1) or (32*32*3) to 1-D
    model.add(Flatten(input_shape=data.train_data.shape[1:]))
    # dense layers (the hidden layer)
    n = 0
    for param in params:
        n += 1
        model.add(Dense(param, kernel_initializer='he_uniform'))
        # ReLU activation
        if activation == "arctan":
            model.add(Lambda(lambda x: tf.atan(x), name=activation+"_"+str(n)))
        else:
            model.add(Activation(activation, name=activation+"_"+str(n)))
    # the output layer, with 10 classes
    model.add(Dense(10, kernel_initializer='he_uniform'))
    
    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    # initiate the SGD optimizer with given hyper parameters
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    
    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.summary()
    print("Traing a {} layer model, saving to {}".format(len(params) + 1, file_name))
    # run training with given dataset, and print progress
    history = model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name)
        print('model saved to ', file_name)
    
    return {'model':model, 'history':history}

if not os.path.isdir('models'):
    os.makedirs('models')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train n-layer MNIST and CIFAR models')
    parser.add_argument('--model', 
                default="mnist",
                choices=["mnist", "cifar"],
                help='model name')
    parser.add_argument('--modelfile', 
                default="",
                help='override the model filename, use user specied one')
    parser.add_argument('--modelpath', 
                default="models_training",
                help='folder for saving trained models')
    parser.add_argument('layer_parameters',
                nargs='+',
                help='number of hidden units per layer')
    parser.add_argument('--activation',
                default="relu",
                choices=["relu", "tanh", "sigmoid", "arctan", "elu", "hard_sigmoid", "softplus"])
    parser.add_argument('--lr',
                default=0.01,
                type=float,
                help='learning rate')
    parser.add_argument('--wd',
                default=1e-5,
                type=float,
                help='weight decay')
    parser.add_argument('--epochs',
                default=50,
                type=int,
                help='number of epochs')
    parser.add_argument('--overwrite',
                action='store_true',
                help='overwrite output file')
    args = parser.parse_args()
    print(args)
    nlayers = len(args.layer_parameters) + 1
    if not args.modelfile:
        file_name = args.modelpath+"/"+args.model+"_"+str(nlayers)+"layer_"+args.activation+"_"+args.layer_parameters[0]
    else:
        file_name = args.modelfile
    print("Model will be saved to", file_name)
    if os.path.isfile(file_name) and not args.overwrite:
        raise RuntimeError("model {} exists.".format(file_name))
    if args.model == "mnist":
        data = MNIST()
    elif args.model == "cifar":
        data = CIFAR()
    train(data, file_name=file_name, params=args.layer_parameters, num_epochs=args.epochs, lr=args.lr, decay=args.wd, activation=args.activation)
    # 2-layer models

    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[10], num_epochs=50, lr=0.03, decay=1e-6)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[50], num_epochs=50, lr=0.05,decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[100], num_epochs=50, lr=0.05, decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[1024], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_2layer_relu", params=[1024], num_epochs=50, lr=0.2, decay=1e-3)
    # 3-layer models
    # train(MNIST(), file_name="models/mnist_3layer_relu", params=[10, 10], num_epochs=50, lr=0.03, decay=1e-7)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[50], num_epochs=50, lr=0.05,decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[100], num_epochs=50, lr=0.05, decay=1e-4)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[1024], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_2layer_relu", params=[1024], num_epochs=50, lr=0.2, decay=1e-3)
    # 3-layer models
    # train(MNIST(), file_name="models/mnist_3layer_relu", params=[10, 10], num_epochs=50, lr=0.03, decay=1e-7)
    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[100], num_epochs=50, lr=0.05, decay=1e-4)

    # train(MNIST(), file_name="models/mnist_2layer_relu", params=[1024], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_2layer_relu", params=[1024], num_epochs=50, lr=0.2, decay=1e-3)
    # 3-layer models
    # train(MNIST(), file_name="models/mnist_3layer_relu_10_10", params=[10, 10], num_epochs=50, lr=0.03, decay=1e-7)
    # train(MNIST(), file_name="models/mnist_3layer_relu", params=[256,256], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_3layer_relu", params=[256,256], num_epochs=50, lr=0.2, decay=1e-3)
    # 4-layer models
    # train(MNIST(), file_name="models/mnist_4layer_relu", params=[256,256,256], num_epochs=50, lr=0.1, decay=1e-3)
    # train(CIFAR(), file_name="models/cifar_4layer_relu", params=[256,256,256], num_epochs=50, lr=0.2, decay=1e-3)
    # train(MNIST(), file_name="models/mnist_4layer_relu", params=[20,20,20], num_epochs=50, lr=0.07, decay=1e-3)
    # train(MNIST(), file_name="models/mnist_5layer_relu", params=[20,20,20,20], num_epochs=50, lr=0.03, decay=1e-4) 
    # train(MNIST(), file_name="models/mnist_5layer_relu", params=[20,20,20,20], num_epochs=50, lr=0.02, decay=1e-4)

