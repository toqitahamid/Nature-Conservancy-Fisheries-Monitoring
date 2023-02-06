from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger, ReduceLROnPlateau
import json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from keras.regularizers import l2

IMGS_DIM_3D = (3, 256, 256)
MAX_EPOCHS = 500
BATCH_SIZE = 96
L2_REG = 0.003
W_INIT = 'he_normal'
LAST_FEATURE_MAPS_LAYER = 46
LAST_FEATURE_MAPS_SIZE = (128, 8, 8)
PENULTIMATE_LAYER = 51
PENULTIMATE_SIZE = 2048
SOFTMAX_LAYER = 55
SOFTMAX_SIZE = 8



def custom_1_model(img_rows, img_cols, channel=1, num_class=None):
    model = Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=(channel, img_rows, img_cols)))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(PReLU(init=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(p=0.5))

    model.add(Flatten())
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    model.add(BatchNormalization(mode=2))
    model.add(PReLU(init=W_INIT))

    
    model.add(Dropout(p=0.5))
    model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
    model.add(BatchNormalization(mode=2))
    model.add(Activation(activation='softmax'))
    
    sgd = SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #adam = Adam(lr=0.000074)
    #model.compile(loss='categorical_crossentropy',  optimizer=adam, metrics=['accuracy'])

    return model
    
def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, input_shape=input_shape,
        border_mode='same', init=W_INIT, W_regularizer=l2(l=L2_REG))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(
        nb_filter=nb_filter, nb_row=3, nb_col=3, border_mode='same',
        init=W_INIT, W_regularizer=l2(l=L2_REG))


def _dense_layer(output_dim):
    return Dense(output_dim=output_dim, W_regularizer=l2(l=L2_REG), init=W_INIT)
  

def load_data():
  """
  Load dataset and split data into training and validation sets
  """
  img_width, img_height = 256, 256

  train_data_dir = '../../../data/train_split'
  validation_data_dir = '../../../data/val_split'
  
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
                shuffle = True,
                classes = FishNames,
                class_mode='categorical')




  validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                shuffle = True,
                classes = FishNames,
                class_mode='categorical')
  
  
  return (train_generator, validation_generator)
  

def load_data_no_crop(FishNames):
  """
  Load dataset and split data into training and validation sets
  """
  img_width, img_height = 256, 256

  train_data_dir = '../../../data/train_no_crop'
  validation_data_dir = '../../../data/validation_no_crop'
  
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

  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                shuffle = True,
                classes = FishNames,
                class_mode='categorical')




  validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                shuffle = True,
                classes = FishNames,
                class_mode='categorical')
  
  
  return (train_generator, validation_generator)
 
  
def plot(history): 
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')           
    plt.legend(['train', 'test'], loc='upper left')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('graph/custom_1_v1_finetune_accuracy.png', dpi=1000)


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig('graph/custom_1_v1_finetune_loss.png', dpi=1000)
     
def save_model_to_json(model):
    print('saving model...')
    with open('model\custom_1_v1.json', 'w') as f:
        f.write(model.to_json())
    print('model saved...')    
    
def save_history_json(history):
    with open('history/custom_1_v1_history.json', mode='w') as f:
        json.dump(history.history, f)


if __name__ == '__main__':
    
    

    # Fine-tune Example
    img_rows, img_cols = 256, 256 # Resolution of inputs
    channel = 3
    num_class = 8 
    batch_size = 16 
    nb_train_samples = 3019
    nb_validation_samples = 758
    nb_epoch = 100
    
    FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    # TODO: Load training and validation sets
    train_generator, validation_generator = load_data()

    # Load our model
    model = custom_1_model(img_rows, img_cols, channel, num_class)
    
    save_model_to_json(model)
    
    # TODO: Start Fine-tuning
    filepath="weight/custom_1_v1_finetune.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
    mode='auto')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=0.0000001) 
    
    csv_logger = CSVLogger('history/custom_1_v1_history.log', append=True)
    finetune_callbacks_list = [checkpoint, csv_logger, reduce_lr] 
        
     
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        verbose=1,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples, 
        callbacks=finetune_callbacks_list)
    
    plot(history)
    save_history_json(history)
    
    
    # Make predictions
    #predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    #score = log_loss(Y_valid, predictions_valid)