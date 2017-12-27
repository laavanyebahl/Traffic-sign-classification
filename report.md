# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"

[image2]: ./examples/before.png "Before Pre processing"
[image3]: ./examples/after.png "After Pre processing"
[image4]: ./examples/web_image.png "Web Test Images"
[image5]: ./examples/web_image_result.png "Web Test Images Result"
[image6]: ./examples/softmax.png "Softmax"
[image7]: ./examples/sample.png "sample"
[image8]: ./examples/prediction.png "prediction"
[image9]: ./examples/top5.png "top5"


---
### Writeup / README

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 image
* The size of the validation set is 4410 image
* The size of test set is 12630 image
* The shape of a traffic sign image is 32x32 
* The number of unique classes/labels in the data set is 43 classes

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

## Data Preprocessing
  
1. Normalizing the data using the OpenCV Library Min-Max Normalization:
      * Reduced the pixels values between -1. and 1. to centralize the data around the origin
      * Reduced the data's mean and standard deviation for better and more valuable training.
     
     Here is an example of a traffic sign image before pre processing.

     ![alt text][image2]             
     
     Here is an example of a traffic sign image after pre processing.
          
     ![alt text][image3]  

2. Shuffling the entire data set, and labels, using the sklearn library:
     * Randomly shuffling the data for training to attain a random distribution throughout each batch for Stochastic Gradient Descent.
     * The shuffling had a major role in the training of the model and a huge impact on the network's accuracy.


## Model Architecture

This model architecture follows the implementation of the LeNet CNN.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 --  RGB image   							| 
| Convolution 5x5     	| 1x1 stride + VALID padding + outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride + outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride + VALID padding + outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride + outputs 5x5x16 				|
| Fully connected		| 120 unit        									|
| RELU					|												|
| Dropout				| 0.5 keep_prob        									|
| Fully connected		| 83 unit        									|
| RELU																	|
| Dropout				| 0.5 keep_prob        									|
|	Output					|	43 Logits											|




## Model Training
### Tuned Parameters
The choice of these hyperparameters required several trials to reach this final combination which represents the maximum performance achieved.
1. Epochs # 40   --------- Chosen upon the fact that the model reaches a plateau by the 15th epoch
2. Batch Size # 64  --------- Appropriate and efficient batch size     
3. Learning Rate = 0.001 --------- Through many tests, this learning rate was right before overshooting, nonetheless fast and converges
4. Mean = 0.  &  Standard Deviation = 0.1 --------- Values fed for the tf.truncated_normal() function for weight initialization 
5. Dropout = 0.5 --------- The probability for the dropout layers which decreased vastly the overfitting of the dataset

    * Used the tf.nn.softmax_cross_entropy_with_logits() function to calculate the logits probabilities using: softmax + the cross entropy 
    * Used the Adam Optimizer for training the network with backpropagation and stochastic gradient descent.


## Model Performance
My final model results were:
* Validation Set Accuracy = 96.07 % 
* Test Set Accuracy = 94.27 %


## Testing the Model on New Images
### Introduction to the Chosen Images
##### These are 7 German traffic signs that I found on the web:

![alt text][image4]

After pre processing

![alt text][image5]


### Prediction Results

![alt text][image8]

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.714 %. This compares favorably to the accuracy on the test set of 94.2 %

### This is a table representing the 7 images tested, and the top 5 softmax probabilities produced by the network analyzing these images:

![alt text][image9]


## Feature Map Visualization of 2nd Convolutional Layer

#### The following image, representing a traffic sign is run through the network.
![alt text][image7]

#### This image activates 16 different Feature Maps in the 2nd Convolutional Layer, representing what did each feature map observe in this image. as visualized below:
![alt text][image6]



