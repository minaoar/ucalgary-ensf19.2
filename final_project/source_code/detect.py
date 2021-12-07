import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

import PIL
from PIL import Image
import numpy as np
import glob
import sys
import datetime
from shutil import copy2


img_height = 224
img_width = 224

if sys.argv[1] == "enb0":
  img_height = 256
  img_width = 256

project_home = "/home/minaoar.tanzil/gender-detection/"
data_dir = project_home+"/detection/"
female_dir = data_dir+"/female/"
male_dir = data_dir+"/male/"


#class_names = ["Male", "Female"]

model_names_ft = {}
model_names_ft["enb0"]=["en_b0", "EfficientNet B0"]
model_names_ft["vgg16"]=["vgg_16", "VGG 16"]
model_names_ft["vgg19"]=["vgg_19", "VGG 19"]
model_names_ft["rn50"]=["rn_50", "Resnet50"]
model_names_ft["rn50v2"]=["rn_50_v2", "Resnet50 V2"]
model_names_ft["mn"]=["mn", "MobileNet"]
model_names_ft["mnv2"]=["mn_v2", "MobileNet V2"]

model_name_ft = project_home+"output/gender_classifier_"+model_names_ft[sys.argv[1]][0]+"_ft.h5"


# load weights
model = tf.keras.models.load_model(model_name_ft)

print("Detecting gender...", data_dir, model_name_ft )

# Data generator parameters
gen_params = {"featurewise_center":False,\
              "samplewise_center":False,\
              "featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,\
              "zca_whitening":False,\
              "horizontal_flip":False,\
              "vertical_flip":False}

# Train and validation generators
detect_gen = ImageDataGenerator(**gen_params)


detect_generator = detect_gen.flow_from_directory(
    directory = data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes= None,
    class_mode=None,
    batch_size=32,
    shuffle=False,
    #seed=42,
    interpolation="nearest"
)
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))

print("Images loaded from detection directory", len(detect_generator.filepaths))


result = model.predict(detect_generator,verbose=1)
print("Detection done!")
print(result.shape)
print(result)
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))

total = result.shape[0]
y_pred = np.zeros(total)
for ii in np.arange(0, total):
  y_pred[ii] = np.argmax(result[ii])

female = np.sum(y_pred)
male =  total - female
print("=======================================")
print("Model used for detection:", model_names_ft[sys.argv[1]][1])
print("Detection completed for faces:", total)
print("Female percentage:", round(female/total*100, 2))
print("Total Female: ", female)
print("Total Male: ", male)
print("=======================================")


f_male = open(data_dir+"male.txt", "a")
f_female = open(data_dir+"female.txt", "a")

print("Listing detected images to text files.")
for jj in range(0, len(detect_generator.filepaths)):
  imageName = detect_generator.filenames[jj]
  if y_pred[jj] == 0:
      f_male.write(imageName+"\n")
  else:
      f_female.write(imageName+"\n")
  if jj % int(total/50) == 0:
    print(">", end='')
print("")

f_male.close()
f_female.close()

print("Copying detected files.")
print("=================================================") 
for jj in range(0, len(detect_generator.filepaths)):
  imagePath = detect_generator.filepaths[jj]
  if y_pred[jj] == 0:
      copy2(imagePath, male_dir)
  else:
      copy2(imagePath, female_dir)

  if jj % int(total/50) == 0:
    print(">", end='')
print("")

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))
print("All files copied.", )
  