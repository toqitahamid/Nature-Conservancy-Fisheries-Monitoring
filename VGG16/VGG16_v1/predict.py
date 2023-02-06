from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger
import json
import theano

def vgg_std16_model(img_rows, img_cols, channel=1, num_class=None):
    """
    VGG 16 Model for Keras
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_class - number of class labels for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

   
    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_class, activation='softmax'))

    
    model.load_weights('weight/VGG16_v3_finetune.best.hdf5')
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



def load_data():
  """
  Load dataset and split data into training and validation sets
  """
  img_width, img_height = 224, 224

  train_data_dir = '../../../data/train'
  validation_data_dir = '../../../data/validation'
  
  train_datagen = ImageDataGenerator(
                rescale=1./255,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=45,
                width_shift_range=0.25,
                height_shift_range=0.25,
                horizontal_flip=True,
                vertical_flip=False,
                zoom_range=0.5,
                channel_shift_range=0.5,
                fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling

  test_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=45,
                                  width_shift_range=0.25,
                                  height_shift_range=0.25,
                                  horizontal_flip=True)

  train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                class_mode='categorical')




  validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                class_mode='categorical')
  
  
  return (train_generator, validation_generator)
  
def test_data_no_crop():
    
  img_width, img_height = 224, 224

  test_data_dir = '../../../data/validation_no_crop'
 
  test_datagen = ImageDataGenerator(rescale=1./255)

  test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                class_mode='categorical')
  
  
  return (test_generator)
  
def test_data():
    
  img_width, img_height = 224, 224

  test_data_dir = '../../../data/validation'
 
  test_datagen = ImageDataGenerator(rescale=1./255)

  test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                class_mode='categorical')
  
  
  return (test_generator)
  
  
def save_model_to_json(model):
    print('saving model... ')
    with open('model\model.json', 'w') as f:
        f.write(model.to_json())
        
def save_history_json(history):
    with open('history/inception_v3_v4_history.json', mode='w') as f:
        json.dump(history.history, f)

def vis_layer(model):
    # visualizing intermediate layers

    X_train = np.load('../../../data/array_data/trainData.npy')
    Y_train = np.load('../../../data/array_data/trainLabels.npy')
    Y_class = np.load('../../../data/array_data/trainClass.npy')
    
    output_layer = model.layers[3].output
    output_fn = theano.function([model.layers[0].input], output_layer)
    
    # the input image
    
    input_image=X_train[0:1,:,:,:]
    print(input_image.shape)
    
    plt.imshow(input_image[0,0,:,:],cmap ='gray')
    plt.imshow(input_image[0,0,:,:])
    
    
    output_image = output_fn(input_image)
    print(output_image.shape)
    
    # Rearrange dimension so we can plot the result 
    output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
    print(output_image.shape)
    
    
    fig=plt.figure(figsize=(224,224))
    for i in range(32):
        ax = fig.add_subplot(6, 6, i+1)
        #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
        ax.imshow(output_image[0,:,:,i],cmap=matplotlib.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt       

def conf_matrix(model):
    X_test = np.load('../../../data/array_data/testData.npy')
    Y_test = np.load('../../../data/array_data/testLabels.npy')
    Y_class = np.load('../../../data/array_data/testClass.npy')
    
    # Confusion Matrix

    from sklearn.metrics import classification_report, confusion_matrix
    
    #Y_pred = model.predict(X_test)
    #print(Y_pred)
    #y_pred = np.argmax(Y_pred, axis=1)
    #print(y_pred)
     
    #(or)
    
    y_pred = model.predict_classes(X_test)
    print(y_pred)
    
    p=model.predict_proba(X_test) # to predict probability
    
    target_names = Y_class
    print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))
    d = confusion_matrix(np.argmax(Y_test,axis=1), y_pred)
     
    e = np.amax(d, axis=1)
    
    a = classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names)

    print(e)
        
if __name__ == '__main__':

    # Fine-tune Example
    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_class = 196 
    batch_size = 16 
    nb_train_samples = 8144
    nb_validation_samples = 8041
    nb_epoch = 100

    # TODO: Load training and validation sets
    test_generator = test_data()

    # Load our model
    model = vgg_std16_model(img_rows, img_cols, channel, num_class)
    '''
    save_model_to_json(model)
    
    scores = model.evaluate_generator(test_generator, val_samples=nb_validation_samples)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    '''
    #conf_matrix(model)
    #vis_layer(model)
    
    
     
   # Y_valid = np.load('testLabels.npy')

    # Make predictions
    #predictions_valid = model.predict_generator(train_generator, samples_per_epoch=nb_train_samples)

    # Cross-entropy loss score
   # score = log_loss(Y_valid, predictions_valid)
