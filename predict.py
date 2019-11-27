from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.mobilenet import preprocess_input
from urllib.request import urlopen
from PIL import Image
import numpy as np
import sys

classes = ["cat", "dog", "horse"]

# load json and create model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

img_name = sys.argv[1]
image = load_img(f"test_imgs/{img_name}", target_size=(224, 224))
x = img_to_array(image)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(f"Image {img_name} loaded!")

prediction = model.predict(x)
print("Predictions:", prediction)

print()
print("This is a", classes[np.argmax(prediction)])
