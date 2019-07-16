# **Behavioral Cloning Project** 
---

In this project:
* I used a simulator to collect data of good driving behavior.
* Built a convolution neural network in Keras that predicts steering angles from images.
* Trained and validated the model with a training and validation set.
* Tested that the model successfully drives around track one without leaving the road.

---

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
[![Video](http://i.imgur.com/5treVet.png)](https://vimeo.com/229864255 "Video")


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 97-115) 

The model includes RELU layers to introduce nonlinearity (code line 105-109), and the data is normalized in the model using a Keras lambda layer (code line 100). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 122-139). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the sample dataset.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use Nvidia Architecture as described here:

http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

I thought this model might be appropriate because it looked perfect for this assignment and also because I think Nvidia is a great company!

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I adjusted the correction for the left and right steering angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 97-115) consisted of a convolution neural network with the following layers and layer sizes:
![](http://i.imgur.com/ZIW6xMz.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first tried to use the sample dataset. As it turned out, the sample dataset had enough data points to make the model work.

Here are the images from the dataset of the center, right and left cameras, when the car is on the bridge:

![Center](http://i.imgur.com/S2OwbLR.jpg)
![Right](http://i.imgur.com/J7IHEWr.jpg)
![Left](http://i.imgur.com/WI4fs9z.jpg)

To augment the dataset, I used the images from all 3 cameras and changed the steering angle accordingly. I used 0.14 correction to the main steering angle.

After the collection process, I had 19284 data points. I then preprocessed this data by trimming , grayscaling and normalizing the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the following image:

![](http://i.imgur.com/bf2jiAV.png)

I used an adam optimizer so that manually training the learning rate wasn't necessary.
