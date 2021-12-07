import tensorflow as tf
print(tf.__version__)
import numpy as np
import glob
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import utils
import datetime

project_home = "/home/minaoar.tanzil/gender-detection/"

model_name_it = project_home+"output/gender_classifier_rn_50_it.h5"
model_name_ft = project_home+"output/gender_classifier_rn_50_ft.h5"

data_dir = project_home+"data/"

class_names = ["Male", "Female"]

img_height_input = 200
img_width_input = 200
img_height = 224
img_width = 224
batch_size = 32



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
              "horizontal_flip":True,\
              "vertical_flip":True}

# Train and validation generators
train_gen = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.resnet.preprocess_input, validation_split=.2)
val_gen = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.resnet.preprocess_input, validation_split=.2)



train_generator = train_gen.flow_from_directory(
    directory = data_dir,
    target_size=(img_height, img_width),
    color_mode="rgb",
    classes= class_names,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=42,
    interpolation="nearest",
    subset="training"
)

validation_generator = val_gen.flow_from_directory(
    directory = data_dir,
    target_size=(img_height, img_width),
    color_mode="rgb",
    classes= class_names,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=42,
    interpolation="nearest",
    subset="validation"
)


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)


monitor_it = tf.keras.callbacks.ModelCheckpoint(model_name_it, monitor='val_loss',\
                                             verbose=0,save_best_only=True,\
                                             save_weights_only=False,\
                                             mode='min')

monitor_ft = tf.keras.callbacks.ModelCheckpoint(model_name_ft, monitor='val_loss',\
                                             verbose=0,save_best_only=True,\
                                             save_weights_only=False,\
                                             mode='min')

def scheduler(epoch, lr):
    if epoch%10 == 0 and epoch!= 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)



# Defining the model
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(img_height, img_width, 3),
    include_top=False) 
base_model.trainable = False

x1 = base_model(base_model.input, training = False)
x2 = tf.keras.layers.Flatten()(x1)
out = tf.keras.layers.Dense(len(class_names),activation = 'softmax')(x2)
model = tf.keras.Model(inputs = base_model.input, outputs =out)

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))

print("Initial Training Model")
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_it = model.fit(train_generator, epochs=100, verbose = 1, \
                       callbacks= [early_stop, monitor_it, lr_schedule], \
                       validation_data = (validation_generator))


# Fine-tuning the model
model = tf.keras.models.load_model(model_name_it)
model.trainable = True

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))


print("Fine-tuning model")
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-8),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history_ft = model.fit(train_generator, epochs=50, verbose = 1, \
                       callbacks= [early_stop, monitor_ft, lr_schedule], \
                       validation_data = (validation_generator))

print("Fine-tuning done!")
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))


