
# coding: utf-8

# In[34]:

# Imports

import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.preprocessing import image
from keras.models import Model,load_model,Sequential
from keras.layers import Lambda,Flatten, Dense,Cropping2D,Convolution2D
from sklearn.model_selection import train_test_split
from keras import backend as K


# In[35]:

def upload_labels(file):
    samples = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


# In[36]:

# Upload the input

lables_file ='./driving_log.csv'
samples = upload_labels(lables_file)


# In[37]:

# Split the samples to train and validation samples

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[38]:

def get_imgs(batch_sample):
    batch_car_imgs =[]
    directory_imgs = './IMG/'
    
    for i in range (3):
        img_file = directory_imgs +batch_sample[i].split('/')[-1]
        img = cv2.imread(img_file)
        batch_car_imgs.append(img)
    
    return batch_car_imgs


# In[47]:

def get_angles(batch_sample):
    correction = 0.14
    steering_center = float(batch_sample[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
                
    new_steering_angles = [steering_center, steering_left, steering_right]
    
    return new_steering_angles


# In[48]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                car_images.extend(get_imgs(batch_sample))
                steering_angles.extend(get_angles(batch_sample))

            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)


# In[49]:

def create_model():
    ch, row, col = 3, 160, 320
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))

    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
   
    return model


# In[50]:

# Run and train the model

def run_model():
    
    model = create_model()
    model.compile(loss='mse', optimizer='adam')

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    history_object = model.fit_generator(train_generator, 
                                         samples_per_epoch =3*len(train_samples), 
                                         validation_data = validation_generator,
                                         nb_val_samples = 3*len(validation_samples), 
                                         nb_epoch=5)
    
    model.save('model.h5')
    
    return history_object


# In[51]:

def print_model(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# In[52]:

def main():
    history_object = run_model()
    print_model(history_object)


# In[53]:

main()

