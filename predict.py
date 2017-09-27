# if you are on 32 bit os
# import Image
import resizeimage
# 64 bit with pillow:
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import os
import camera
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pickle



classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()



# load json and create model
json_file = open('model_face.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_face.h5")
print("Loaded model from disk")

while True:

 try:
  image=np.array([camera.get_image()])
  image = image.astype('float32')
  image = image / 255.0

  prediction=loaded_model.predict(image)

  print(prediction)

  print(np.max(prediction))

  if(np.max(prediction)>0.7):

     print(int_to_word_out[np.argmax(prediction)])

  else:

     print("face_not_found")


 except resizeimage.imageexceptions.ImageSizeError:
   print("1")


 except KeyboardInterrupt:
  exit()






