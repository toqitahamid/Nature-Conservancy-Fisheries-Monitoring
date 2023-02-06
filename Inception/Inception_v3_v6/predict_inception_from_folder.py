from keras.applications.resnet50 import ResNet50
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import cv2
import os
import glob
from keras.layers import Input
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image as image_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

img_width, img_height = 299, 299
num_classes = 196

def get_inception():
     # Load our model
    json_file = open('model/inception_v3_v3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("weight/inception_v3_v3_finetune.best.hdf5")
    print("Loaded model from disk")
    return model


def read_img(img_path):
    """
    this function returns preprocessed image
    """
    dim_ordering = K.image_dim_ordering()
    mean = (103.939, 116.779, 123.68)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img, dim_ordering=dim_ordering)

    # decenterize
    #img[0, :, :] -= mean[0]
    #[1, :, :] -= mean[1]
    #img[2, :, :] -= mean[2]

    # 'RGB'->'BGR'
    if dim_ordering == 'th':
        img = img[::-1, :, :]
    else:
        img = img[:, :, ::-1]

    # expand dim for test
    img = np.expand_dims(img, axis=0)
    return img

def read_data():
    print("reading test data")

    datadir = '../../../data/search'
    subfiles = os.listdir(datadir)
    print subfiles
    
    i_class = 0
    data = []
    labels = []
    ilabels = []
    klabels = []
    


    for i_file in subfiles:
        i_file_path = datadir + '/' + i_file
        if os.path.isdir(i_file_path):
            jpgfiles = glob.glob(i_file_path + '/' + '*.jpg')
            for j_file in jpgfiles:
                image = image_utils.load_img(j_file, target_size=(img_width, img_height))
                image = image_utils.img_to_array(image)
                label = i_file
                data.append(image)
                labels.append(label)                
                
                i_class = i_class + 1
    
    
   
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    ilabels = le.classes_
    vLabels = labels
    labels = np_utils.to_categorical(labels, num_classes)
    
    data = np.array(data) / 255.0         
                    
    testData = data
    testLabels = labels  
    return testData                            

if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    #weights_file = K.image_dim_ordering() + 'ResNet50_finetune.best.hdf5'
    
    resnet_model = get_inception()
    #resnet_model.load_weights('ResNet50_finetune.best.hdf5')
    testData = read_data()
    #test_img1 = read_img('00128.jpg')
    #test_img2 = read_img('00128.jpg')
    # you may download synset_words from address given at the begining of this file
    class_table = open('../../../data/synset_words.txt', 'r')
    lines = class_table.readlines()
    print "result for test 1 is"
    print lines[np.argmax(resnet_model.predict(testData[np.newaxis, 0]))]
    print "result for test 2 is"
    #print lines[np.argmax(resnet_model.predict(testData[np.newaxis, 1]))]
    print "result for test 2 is"
    #print lines[np.argmax(resnet_model.predict(testData[np.newaxis, 2]))]
    class_table.close()
    
    preds = resnet_model.predict(testData[np.newaxis, 0])
    
    
    for pred in preds:
        top_indices = pred.argsort()[-5:][::-1]
        for i in top_indices:
            a= i
            print('I am {:.2%} sure this is a {}'.format(pred[i], str(lines[a])))
