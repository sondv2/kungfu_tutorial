import pickle
from sklearn.model_selection import train_test_split
from scipy import misc
import numpy as np
import os

#load dataset
label = os.listdir("dataset_image")

def load_datasets():

    dataset=[]
    for image_label in label:

        images = os.listdir("dataset_image/"+image_label)

        for image in images:
            img = misc.imread("dataset_image/"+image_label+"/"+image)
            img = misc.imresize(img, (64, 64))
            dataset.append((img,image_label))

    X=[]
    y=[]
    for  input,image_label in dataset:

        X.append(input)

        y.append(label.index(image_label))

    X=np.array(X)
    y=np.array(y)


    return X,y



save_label = open("int_to_word_out.pickle","wb")
pickle.dump(label, save_label)
save_label.close()
