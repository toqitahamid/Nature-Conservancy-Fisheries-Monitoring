# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:10:47 2016

@author: Prithi
"""

from keras.models import load_model, model_from_json
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.hdf5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()    

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000
nbr_augmentation = 50

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

#root_path = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'
root_path = 'result'
weights_path= 'weight/second_network_finetune.best.hdf5'

test_data_dir = '../../../data/test_stg1/'

# test data generator for prediction
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)


num_fold = 3
nfolds = 3
modelStr = '_inception_v3_16_2x20'

for index in range(1, num_fold + 1):
    InceptionV3_model = read_model(index, modelStr)
    
    print('Loading model and weights from training process ...')


    for idx in range(nbr_augmentation):
        print('{}th augmentation for testing ...'.format(idx))
        random_seed = np.random.random_integers(0, 100000)
    
        test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                shuffle = False, # Important !!!
                seed = random_seed,
                classes = None,
                class_mode = None)
    
        test_image_list = test_generator.filenames
        #print('image_list: {}'.format(test_image_list[:10]))
        print('Begin to predict for testing data ...')
        if idx == 0:
            predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
        else:
            predictions += InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
    
    predictions /= nbr_augmentation
    
    if index == 1:
        yfull_test = predictions
    else:
        yfull_test += predictions

predictions /= nfolds

print('Begin to write submission file ..')
f_submit = open('result/submit.csv', 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
