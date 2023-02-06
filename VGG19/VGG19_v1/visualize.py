from keras.models import model_from_json
from quiver_engine import server
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19



'''
json_file = open('model\model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("weight\VGG19_v1_finetune.best.hdf5")
print("Loaded model from disk")
'''

model = ResNet50(weights='imagenet')

server.launch(
        model, # a Keras Model

        # where to store temporary files generatedby quiver (e.g. image files of layers)
        temp_folder='tmp/',

        # a folder where input images are stored
        input_folder='img/',

        # the localhost port the dashboard is to be served on
        port=3000
    )
