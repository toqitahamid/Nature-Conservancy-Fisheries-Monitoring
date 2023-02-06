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
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger, ReduceLROnPlateau, TensorBoard
import json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2
from keras.metrics import top_k_categorical_accuracy, precision, recall, fbeta_score, f1score
import os






def inception_v3_model(img_rows, img_cols, channel=1, num_class=None):
    K.clear_session()

    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_rows, img_cols, channel)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.8)(x)
    x = Flatten()(x)
    predictions = Dense(num_class, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)
    
    model = Model(input=base_model.input, output=predictions)
    #model.load_weights('weight/inception_v3_v7_tf.hdf5')
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', top_k_categorical_accuracy, precision, recall, f1score])

    return model 

def inception_preprocess(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def load_data():
  """
  Load dataset and split data into training and validation sets
  """
  img_width, img_height = 299, 299

  train_data_dir = '../../../data/train_split'
  validation_data_dir = '../../../data/val_split'
  
  train_datagen = ImageDataGenerator(
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
                fill_mode='nearest',
                preprocessing_function=inception_preprocess
                )

# this is the augmentation configuration we will use for testing:
# only rescaling

  test_datagen = ImageDataGenerator(
                                  preprocessing_function=inception_preprocess
                                  )

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
  

def load_data_no_crop():
  """
  Load dataset and split data into training and validation sets
  """
  img_width, img_height = 299, 299

  train_data_dir = '../../../data/train_no_crop'
  validation_data_dir = '../../../data/validation_no_crop'
  
  train_datagen = ImageDataGenerator(
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
                fill_mode='nearest',
                preprocessing_function=inception_preprocess)

# this is the augmentation configuration we will use for testing:
# only rescaling

  test_datagen = ImageDataGenerator(
          preprocessing_function=inception_preprocess
          )

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
 
  
def plot(history, version, backend): 
    
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
    fig1.savefig('graph/inception_v3_v'+ version +'_'+ backend +'_accuracy.png', dpi=1000)


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig('graph/inception_v3_v'+ version +'_'+ backend +'_loss.png', dpi=1000)
     
def save_model_to_json(model, version, backend):
    print('saving model...')
    with open('model\inception_v3_v'+ version +'_'+ backend +'.json', 'w') as f:
        f.write(model.to_json())
    print('model saved...')    
    
def save_history_json(history, version, backend):
    with open('history/inception_v3_v'+ version +'_'+ backend +'.json', mode='w') as f:
        json.dump(history.history, f)




if __name__ == '__main__':
    
    version = '8'
    backend = 'tf'
    
    # Fine-tune Example
    img_rows, img_cols = 299, 299 # Resolution of inputs
    channel = 3
    num_class = 8 
    batch_size = 16 
    nb_train_samples = 3019
    nb_validation_samples = 758
    nb_epoch = 200
    
    # TODO: Load training and validation sets
    train_generator, validation_generator = load_data()

    # Load our model
    model = inception_v3_model(img_rows, img_cols, channel, num_class)
    
    save_model_to_json(model, version, backend)
    
    # TODO: Start Fine-tuning
    filepath='weight/inception_v3_v'+ version +'_'+ backend +'.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
    mode='min')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-20) 
    
    csv_logger = CSVLogger('history/inception_v3_v'+ version +'_'+ backend +'.log', append=True)
    
    from datetime import datetime
    now = datetime.now()
    #logdir = "tf_logs/.../" + now.strftime("%Y%m%d-%H%M%S") + "/"
    print(now.strftime("%Y%m%d-%H%M%S"))
    logdir = "/logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    print(logdir)
   # os.mkdir(logdir)
    tensorboard = TensorBoard(log_dir=logdir)
   
    finetune_callbacks_list = [checkpoint, csv_logger, reduce_lr, tensorboard] 
    
        
     
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        verbose=1,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples, 
        callbacks=finetune_callbacks_list)
    
    plot(history, version, backend)
    save_history_json(history, version, backend)
    
    
    # Make predictions
    #predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    #score = log_loss(Y_valid, predictions_valid)