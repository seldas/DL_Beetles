import keras
import numpy as np
import os
import h5py
from keras.optimizers import Adam, Adadelta, SGD
from keras.utils import to_categorical

from scripts.Beetles_utils import *

myargs = getopts(argv)

curr_folder = 'Beetle_datamatrix/crop_img/val_1/' # by default use the first validation set.
if '-i' in myargs:
	curr_folder = myargs['-i']

output_file = 'my_model_1.h5' # by default, save the model to 'my_model.h5'. It may overwrite previous model.
if '-o' in myargs:
	output_file = myargs['-o']

opt='sgd'
if '-opt' in myargs:
	opt = myargs['-opt']
	opt_1 = opt
	if opt == 'sgd_custom':
		opt = SGD(lr=0.001)
	
	
dataset = h5py.File(curr_folder+"/dataset_500.h5", "r")
x_train = np.array(dataset['x_train'][:])
x_test = np.array(dataset['x_test'][:])
y_train = np.array(dataset['y_train'][:])
y_test = np.array(dataset['y_test'][:])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

# model = bt.conv_model((224,224,3))
model_new = dense_model((7,7,512))
model_new.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["categorical_accuracy"])

print('training on '+ curr_folder + ' | ' + opt_1 + ' | ' + 'output to: ' + output_file)
model_new.fit(x = x_train, y = y_train, epochs = 1000, batch_size = 128, validation_data=(x_test, y_test))

model_new.save(curr_folder+'/'+output_file) 


