from scripts.Beetles_utils import *
import os
# bt.data_gen("crop_img", 'train', sub_image = True)
# bt.data_gen("crop_img", 'test', sub_image=False)

myargs = getopts(argv)
output_folder = 'Beetle_datamatrix/no_crop_img'
if '-i' in myargs:
	output_folder = myargs['-i']
	
val = 'val_1'
if '-val' in myargs:
	val = myargs['-val']

if not os.path.exists(output_folder+'/'+val):
	os.makedirs(output_folder+'/'+val)

dataset_generate_rand(output_folder, output_folder+'/'+val, train_size=500, test_used=True)