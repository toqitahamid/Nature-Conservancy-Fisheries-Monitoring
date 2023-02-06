from keras.models import model_from_json
import numpy as np
import cv2


def show_image():
    
    testData = np.load('../../../data/inc_data/testDatas.npy')
    testLabels = np.load('../../../data/array_data/testLabels.npy')     
    ilabels = np.load('../../../data/array_data/testClass.npy')
   # vLabels = np.load('../../../data/array_data/svmTestLabels.npy')
        # randomly select a few testing digits
    for i in np.random.choice(np.arange(0, len(testLabels)), size=(50,)):
        # classify the digit
        probs = model.predict(testData[np.newaxis, i] / 255.0)
        prediction = probs.argmax(axis=1)
    
        percentage = np.amax(probs)
        print('I am {:.2%} sure this is a {}'.format(np.amax(probs), str(ilabels[prediction[0]])))
    
        # resize the image from a 28 x 28 image to a 96 x 96 image so we
        # can better see it
        image = (testData[i][0]).astype("uint8")

        #image = (testData[i][0] * 255).astype("uint8")
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, str(ilabels[prediction[0]]), (5, 20),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
     
     
     
    	# show the image and prediction
        print("[INFO] Predicted: {}, Actual: {}".format(ilabels[prediction[0]],
    		ilabels[np.argmax(testLabels[i])]))
        
        
        
        for pred in probs:
            top_indices = pred.argsort()[-5:][::-1]
            a= 0
            for i in top_indices:
                print('I am {:.2%} sure this is a {}'.format(pred[i], str(ilabels[top_indices[a]])))
                a += 1
                
        cv2.imshow("CAR", image)
       # plt.imshow(image)
        cv2.waitKey(0)
        
if __name__ == '__main__':

    # Fine-tune Example
    img_rows, img_cols = 299, 299 # Resolution of inputs
    channel = 3
    num_class = 196 
    batch_size = 16 
    nb_train_samples = 8144
    nb_validation_samples = 8041
    nb_epoch = 0

    # Load our model
    json_file = open('model/inception_v3_v3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("weight/inception_v3_v3_finetune.best.hdf5")
    print("Loaded model from disk")
    #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #plot(history_2)
    show_image()