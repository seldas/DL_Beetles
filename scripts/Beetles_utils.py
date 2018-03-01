import keras
import numpy as np
import h5py
import os
from keras.preprocessing import image
from dl_func.imagenet_utils import preprocess_input, decode_predictions
from keras import layers
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.models import Model
from dl_func.vgg16 import VGG16
from keras.utils import to_categorical
import random
from sys import argv

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def load_dataset(train_file='dataset/train_data.h5', test_file='dataset/test_data.h5'):
	train_dataset = h5py.File(train_file, "r")
	train_set_x_orig = np.array(train_dataset["train_set_x"][:])
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])
	
	classes = np.array(test_dataset["list_classes"][:])
	
	test_dataset  = h5py.File(test_file, "r")
	test_set_x_orig  = np.array(test_dataset["test_set_x"][:])
	test_set_y_orig  = np.array(test_dataset["test_set_y"][:])
	
	train_set_y_orig = train_set_x_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig  = test_set_x_orig.reshape((1, test_set_y_orig.shape[0]))
	
	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def data_gen(dataset_name = 'crop_img_resize', dataset_type = 'train', sub_image = True):
	
	mypath = 'Beetle Images data/'+dataset_name+'/'+dataset_type+'/'
	for (dirpath, dirnames, filenames) in os.walk(mypath):
		for sub_dir in dirnames:
			label_y = sub_dir
			i=0
			total_img = 5000
			x_fin = np.zeros((total_img, 224, 224, 3))
			y_fin = np.zeros((total_img, 1))
			label_fin=[]
			
			output_folder = 'Beetle_datamatrix/'+dataset_name+'/'+dataset_type+'/'
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)
			
			for (dirpath_2, dirnames_2, filenames_2) in os.walk(mypath+label_y):
				for filename in filenames_2:
					if i < total_img:
						img_path = dirpath_2 + '/' + filename
						# print(full_path)
						img = image.load_img(img_path, target_size=(224, 224))
						x = image.img_to_array(img)
						x = np.expand_dims(x, axis=0)
						x = preprocess_input(x)
						x_fin[i,:]=x
						y_fin[i,:]=int(label_y)
						if sub_image ==True:
							name_array = filename.split('_')
							super_image_id =name_array[1]
							label_fin.append(super_image_id)
						#if x_fin.shape[0]==0 :
						#	x_fin = x
						#else:
						#	x_fin = np.concatenate((x_fin, x), axis=0)
						#y_fin.append(label_y)
						i+=1
			#only keep non-zero samples		
			x_fin = x_fin[0:i,:]
			y_fin = y_fin[0:i,:]
			print("Finished "+ label_y + ": " + str(x_fin.shape[0]))

			# y_orig = keras.utils.to_categorical(y_fin, num_classes=None)
			# print(y_orig.shape)
			super_image = list(set(label_fin))
			if sub_image == True:
				for img in super_image:
					used_img = [i for i, x in enumerate(label_fin) if x == img]
					x_img = x_fin[used_img,:]
					y_img = y_fin[used_img,:]
					perm_num = np.random.permutation(x_img.shape[0])
					x = x_img[perm_num,:]
					y = y_img[perm_num,]

					output_file = output_folder+dataset_name+'_'+dataset_type+'_'+label_y+'_'+img+'.h5'
					f = h5py.File(output_file, 'w')
					f.create_dataset("train_set_x", data=x)
					f.create_dataset("train_set_y", data=y)
			else:
				perm_num = np.random.permutation(x_fin.shape[0])
				x = x_fin[perm_num,:]
				y = y_fin[perm_num,]

				output_file = output_folder+dataset_name+'_'+dataset_type+'_'+label_y+'.h5'
				f = h5py.File(output_file, 'w')
				f.create_dataset("train_set_x", data=x)
				f.create_dataset("train_set_y", data=y)

def read_data(data_folder = 'Beetle_dataset/crop_img/train', size=100):
	
	x_train = np.array([])
	y_train = np.array([])
	
	for (dirpath, dirnames, filenames) in os.walk(data_folder):
		for file in filenames:
			if file.endswith('.h5'):
				dataset = h5py.File(dirpath+'/'+file, "r")
				x_train_orig = np.array(dataset['train_set_x'][:])
				y_train_orig = np.array(dataset['train_set_y'][:])
				
				rand_sample = np.random.permutation(x_train_orig.shape[0])[0:min(size, x_train_orig.shape[0])]
				if (x_train.shape[0]==0):
					x_train = x_train_orig[rand_sample,:]
					y_train = y_train_orig[rand_sample,]
				else:
					x_train = np.concatenate((x_train, x_train_orig[rand_sample,:]), axis=0)
					y_train = np.concatenate((y_train, y_train_orig[rand_sample,]), axis=0)
				#print(x_train.shape)
				#print(y_train.shape)
	
	perm_num = np.random.permutation(x_train.shape[0])
	x_train = x_train[perm_num,]
	y_train = y_train[perm_num,]
	return x_train, y_train	


def vgg16_feature(x_fin):
	model = VGG16(weights='imagenet', include_top=False)
	features = model.predict(x_fin)
	return features
	

def dataset_generate(folder, output_folder, train_size=500, test_size=500):
	folder_train = folder + '/train'
	x_train, y_train = read_data(folder_train, train_size)
	y_train = to_categorical(y_train-1, num_classes=15)
	x_train = x_train/255
	x_train = vgg16_feature(x_train)

	folder_test = folder + '/test'
	x_test, y_test = read_data(folder_test, test_size)
	y_test = to_categorical(y_test-1, num_classes=15)
	x_test = x_test/255
	x_test = vgg16_feature(x_test)
	
	file = h5py.File(output_folder+'/dataset_'+str(train_size)+'.h5', 'w')
	file.create_dataset("x_train", data = x_train)
	file.create_dataset("y_train", data = y_train)
	file.create_dataset("x_test", data = x_test)
	file.create_dataset("y_test", data = y_test)


def conv_model(input_shape):
	X_input = Input(input_shape)
	X = Conv2D(16, (3,3), padding='same',kernel_initializer=glorot_uniform(seed=0))(X_input)
	X = BatchNormalization(axis = 3)(X)
	X = Activation("relu")(X)
	X = MaxPooling2D((2,2))(X)
	X = Conv2D(32, (3,3), padding='same')(X)
	X = BatchNormalization(axis = 3)(X)
	X = Activation("relu")(X)
	X = MaxPooling2D((2,2))(X)
	X = Conv2D(64, (3,3), padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3)(X)
	X = Activation("relu")(X)
	X = MaxPooling2D((2,2))(X)
	X = Flatten()(X)
	# X = Dense(1024, activation='relu', name="fc1")(X)
	# X = Dropout(0.5)(X)
	# X = Dense(512, activation='relu', name="fc2")(X)
	# X = Dropout(0.8)(X)
	# X = Dense(256, activation='relu', name="fc3")(X)
	# X = Dense(256, activation='relu', name="fc4")(X)
	X = Dense(16, name="fc_fin")(X)
	X = Activation('softmax')(X)

	model = Model(inputs= X_input, output = X, name='Conv_Model')

	return model


	
	
def dense_model(input_shape):
	X_input = Input(input_shape)
	X = Flatten()(X_input)
	X = Dense(1024, activation='relu', name="fc1")(X)
	X = Dropout(0.8)(X)
	X = Dense(512, activation='relu', name="fc2")(X)
	X = Dense(256, activation='relu', name="fc3")(X)
	X = Dense(128, activation='relu', name="fc4")(X)
	# X = Dropout(0.8)(X)
	# X = Dense(256, activation='relu', name="fc3")(X)
	X = Dense(15, name="fc_fin")(X)
	X = Activation('softmax')(X)

	model = Model(inputs= X_input, output = X, name='Dense_Model')

	return model
	
def dataset_generate_rand(folder, output_folder, train_size=500, test_used=False):
	total_file=[]
	with open(folder+"/sample_group.txt", 'r') as file:
		for line in file:
			content = line.split("\t")
			if test_used==True:
				total_file.append([folder+'/'+content[0]+'/'+content[1], content[2]])
			elif test_used==False:
				if content[0] == 'train':
					total_file.append([folder+'/'+content[0]+'/'+content[1], content[2]])
	random.shuffle(total_file)
	
	train_file = []
	test_file  = []
	used_class = []
	for name, classes in total_file:
		if classes not in used_class:
			test_file.append(name)
			used_class.append(classes)
		else:
			train_file.append(name)

	x_train, y_train = read_data_from_list(train_file, train_size)
	y_train = to_categorical(y_train-1, num_classes=15)
	x_train = x_train/255
	x_train = vgg16_feature(x_train)

	folder_test = folder + '/test'
	x_test, y_test = read_data_from_list(test_file, train_size)
	y_test = to_categorical(y_test-1, num_classes=15)
	x_test = x_test/255
	x_test = vgg16_feature(x_test)
	
	file = h5py.File(output_folder+'/dataset_'+str(train_size)+'.h5', 'w')
	file.create_dataset("x_train", data = x_train)
	file.create_dataset("y_train", data = y_train)
	file.create_dataset("x_test", data = x_test)
	file.create_dataset("y_test", data = y_test)


def read_data_from_list(file_list, size=100):
	
	x = np.array([])
	y = np.array([])
	
	for file in file_list:
		if file.endswith('.h5'):
			dataset = h5py.File(file, "r")
			x_orig = np.array(dataset['train_set_x'][:])
			y_orig = np.array(dataset['train_set_y'][:])
			
			rand_sample = np.random.permutation(x_orig.shape[0])[0:min(size, x_orig.shape[0])]
			if (x.shape[0]==0):
				x = x_orig[rand_sample,:]
				y = y_orig[rand_sample,]
			else:
				x = np.concatenate((x, x_orig[rand_sample,:]), axis=0)
				y = np.concatenate((y, y_orig[rand_sample,]), axis=0)
	
	
	perm_num = np.random.permutation(x.shape[0])
	x = x[perm_num,]
	y = y[perm_num,]
	return x, y