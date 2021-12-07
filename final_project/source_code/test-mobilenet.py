import tensorflow as tf
print(tf.__version__)
import numpy as np
import PIL
from PIL import Image
import glob
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0


img_height = 224
img_width = 224

project_home = "/home/minaoar.tanzil/gender-detection/"

model_name_it = project_home+"output/gender_classifier_mn_it.h5"
model_name_ft = project_home+"output/gender_classifier_mn_ft.h5"

data_dir = project_home+"data/"

#imgPath = project_home+"test/27_0_0_20170116213542860.jpg.chip.jpg"
#imgPath = project_home+"test/Female/github-23132028.jpg"
#imgPath = project_home+"test/Female/github-29049143.jpg"
imgPath = project_home+"test/Female/woman_16.jpg"
print(imgPath)

class_names = ["Male", "Female"]

def read_preprocess_image(imgPath,img_width,img_height):
      
    img = load_img(imgPath,target_size=(img_width,img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1, img_width, img_height,3)
    #imgArray = imgArray/float(255)
    np.array([ imgArray ])    
    return imgArray




# read and Pre-processing image
img = read_preprocess_image(imgPath,img_width,img_height)

# load weights
model = tf.keras.models.load_model(model_name_ft)



#images = []
#for f in glob.iglob(project_home+"test/Female/*"):
#    images.append(np.asarray(Image.open(f)))
#images = np.array(images)




outLabel = model.predict(img,verbose=1)
print(outLabel)
prediction = np.argmax(outLabel[0])
print(class_names[prediction]) 

print("Evaluating results from test folder...");

# Data generator parameters
gen_params = {"featurewise_center":False,\
              "samplewise_center":False,\
              "featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,\
              "zca_whitening":False,\
              "rotation_range":20,\
              "width_shift_range":0.1,\
              "height_shift_range":0.1, \
              "shear_range":0.2, \
              "zoom_range":0.1,\
              "horizontal_flip":False,\
              "vertical_flip":False}

# Train and validation generators
test_gen = ImageDataGenerator(**gen_params)


test_generator = test_gen.flow_from_directory(
    directory = project_home+"/test/",
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes= class_names,
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
    seed=42,
    interpolation="nearest"
)

print("Image loaded from test directory")


result = model.predict(test_generator,verbose=1)
print("Prediction done!")

print(result.shape)
print(result)

class_a = 17678
class_b = 9489
total = class_a + class_b

y_true = list(np.zeros(class_a)) + list(np.ones(class_b))
y_pred = np.zeros(total)
for ii in np.arange(0, total):
  y_pred[ii] = np.argmax(result[ii])

print(y_pred)
correct = (y_true == y_pred)
print(correct)
accuracy = np.sum(correct)/total
print("Prediction done! Accuracy:", round(accuracy*100, 2))


