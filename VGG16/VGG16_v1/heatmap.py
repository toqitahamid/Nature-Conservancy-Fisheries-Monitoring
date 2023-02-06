from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids

im = preprocess_image_batch(['example/00130.jpg'], img_size=256, crop_size=224, color_mode="bgr")

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('vgg_16',weights_path="../../../cache/vgg16_weights.h5", heatmap=True)
model.compile(optimizer=sgd, loss='mse')

out = model.predict(im)

#s = "n02084071"
#ids = synset_to_dfs_ids(s)
#heatmap = out[0,ids].sum(axis=0)

# Then, we can get the image
import matplotlib.pyplot as plt
plt.imsave("heatmap_dog.png",out)