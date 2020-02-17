from keras.models import Model, load_model
import numpy as np
import cv2
from matplotlib import pyplot as plt

def visualise(model, image):
	layers_outputs = [layer.output for layer in model.layers]
	activation_model = Model(inputs = model.input, outputs = layers_outputs)
	activations = activation_model.predict(image)
	h = 1
	for layer in activations:
		if len(layer.shape) != 4:
			break
		plt.figure(figsize = (10, 4))
		for channel in range(layer.shape[-1]):
			channel_image = layer[0, :, :, channel]
			plt.subplot(4, layer.shape[-1] // 4, channel + 1)
			plt.imshow(channel_image)
			plt.axis('off')
		plt.show()
		h += 1


# testing the funtion
model_path = '/Users/savyakshat/Downloads/weights_30_30_.h5'
model = load_model(model_path)
image_path = '/Users/savyakshat/Downloads/test.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
image = cv2.resize(image, (30, 30))
image = image.reshape((1, image.shape[0], image.shape[1], 1))
plt.imshow(image[0, :, :, 0])
plt.show()

visualise(model, image)
