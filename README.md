## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[softmax_prob_5]:./output_images/softmax_prob_5.png  "Softmax Probability 5"
[dataexploration]: ./output_images/data_exploration_output.png "Dataset Exploration"
[data_distribution_1]: ./output_images/data_distribution_1.png "Data Distribution before augmentation"
[image_rotation]: ./output_images/image_rotation.png "Image Rotation Example"
[image_translation]: ./output_images/image_translation.png "Image Translation Example"
[image_warping]: ./output_images/image_warping.png "Image Warping Example"
[image_scaling]: ./output_images/image_scaling.png "Image Scaling Example"
[image_brightness]: ./output_images/image_brightness.png "Image Brightness Example"
[data_distribution_2]: ./output_images/data_distribution_2.png "Data Distribution after augmentation"
[sample_image_1]:./sample_images/image1.jpg "Yield Sign"
[sample_image_2]:./sample_images/image2.jpg "Keep right sign"
[sample_image_3]:./sample_images/image3.jpg "30 speed limit sign"
[sample_image_4]:./sample_images/image4.jpg "Turn left ahead sign"
[sample_image_5]:./sample_images/image5.jpg "Stop sign"
[sample_output_image_3]:./output_images/30_speed_limit.png "30 speed limit resized sign"
[softmax_prob_1]:./output_images/softmax_prob_1.png "Softmax Probability 1"
[softmax_prob_2]:./output_images/softmax_prob_2.png "Softmax Probability 2"
[softmax_prob_3]:./output_images/softmax_prob_3.png "Softmax Probability 3"
[softmax_prob_4]:./output_images/softmax_prob_4.png "Softmax Probability 4"

---
### README

***1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.**

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

***Data Set Summary & Exploration***

I used the pickle library to load the dataset picle files and loaded output pandas dataframe to proper train, validation and test variables.

This is the quick summary statistics of the traffic
signs data set:

* The size of training set before data augmentation is 34799
* The size of training set after data augmentation is 54299
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

***2. Include an exploratory visualization of the dataset***

Here is an exploratory visualization of the data set. It is a matplot chart showing one image from each distinct class.

![alt text][dataexploration]

***Design and Test a Model Architecture***

***1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)***

Before performing image preprocessing, I noticed that the data distribution across the labels are not uniform.

![alt text][data_distribution_1]

I learned that data augmentation is proper way to create different images with slight modifications and use this approach to fill the data distribution gap between labels. I also learned that the model would also benefit from this step as it learns to understand the signs at different brightness, angles etc.

These are the set of image manipulations I have done as part of data augmentation:
* Apply random rotation to the image
![alt text][image_rotation]
* Apply random translation
![alt text][image_translation]
* Apply slight warping
![alt text][image_warping]
* Scale the image
![alt text][image_scaling]
* Adjust Brightness
![alt text][image_brightness]

Data augmentation step performs all the above the mentioned steps for the input image.

For data augmentation, I only selected images for classes that have data samples less than 750. After augmenting each image, i add them back to the training set.

The data distribution with training dataset after data augmentation looked like this.

![alt text][data_distribution_2]

As a last step, I shuffles the data set to prevent predictability. I then performed grayscale and normalization on all the images in training, validation and test set.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5x1x6     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|				outputs 28x28x6								|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 16x16x6 				|
| Convolution 5x5x6x16     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|				outputs 10x10x16								|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 5x5x16 	
| Fully connected	400	| Output = 120        									|
| RELU					|				outputs 120							|
| Dropout	      	| 0.65
| Fully connected	120	| Output = 84        									|
| RELU					|				outputs 84							|
| Dropout	      	| 0.65	|
| Fully connected	84	| Output = 43        									|

***3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.***

To train the model, I did the following mentioned steps for each training data set input:
* Input each image from the training set to CNN
* Take the output logits and run softmax_cross_entropy_with_logits function with logits and one hot encoded input labels
* Take the overall loss and use the adam optimizer to correct the weights within the CNN

***4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.***

These are some of the steps that helped me achieve a validation set accuracy above 0.93, :
* Used Dropout probability of 0.65 during training
* Performed Data augmentation(rotation, translation, brightness, warping), grayscaling and normalization
* Increased the EPOCHS to 50
* Generated more training samples through data augmentation
* Kept learning rate to 0.0001

My final model results were:
* validation set accuracy of 0.968
* test set accuracy of 0.944
* sample image set accuracy of 0.800

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? I initially started with network of 3 channel input. I didn't include both grayscaling and normalization.
* What were some problems with the initial architecture? the validation accuracy was very bad and below 0.9%
* How was the architecture adjusted and why was it adjusted? I first introduced image preprocessing where i converted images to grayscale and then performed normalization. This improved the accuracy to around 0.9. After that I have introduced dropout and increased the epochs which helped increase the accuracy to about 0.93. Later, i performed data augmentation steps which helped the accuracy to 0.97

* Which parameters were tuned? How were they adjusted and why? Dropout probability, epochs. I started with dropout probability of 0.5 and increased it to 0.65 for better accuracy results.
* What are some of the important design choices and why were they chosen? I felt the data augmentation step helped the architecture a lot. It helped the classifier learns images with different rotation, brightness etc. This step also helped reduce the data distribution inconsistencies across classes

If a well known architecture was chosen:
* What architecture was chosen? Lenet
* Why did you believe it would be relevant to the traffic sign application? Lenet architecture enables us to train a model with traffic sign datasets and help it correct it self during the training process
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?I used the sample random images from the internet and noticed the softmax probabilities are pretty accurate


***Test a Model on New Images***

***1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.***

Here are five German traffic signs that I found on the web:

![alt text][sample_image_1] ![alt text][sample_image_2] ![alt text][sample_image_3]
![alt text][sample_image_4] ![alt text][sample_image_5]

Although, the model was predict most of the signs. I noticed that the image is loosing the valuable pixels while resizing them to 32x32. I might have to see if i could improve this. This is the distorted "30 speed limit sign" image after resizing it to 32x32
![alt text][sample_output_image_3]

***2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).***

Here are the results of the prediction:

| Image			        |     Prediction probability        					|
|:---------------------:|:---------------------------------------------:|
| Yield      		| Yield   								
| Keep right     			| Keep right 									|
| Priority road					| Priority road											|
| 30 km/h	      		| Turn left ahead	|			 		
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.95.

The model predicted correct signs even the resized images were distorted due to resizing and watermarks. I felt, it would've achieved 100% accuracy if it is fed with images without pixelations.

***3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)***

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign. The top five soft max probabilities were shown in the below plot.

![alt text][softmax_prob_1]


For the second image, the model is sure that this is a Keep right sign (probability of 1.0), and the image does contain a Keep right sign. The top five soft max probabilities were shown in the below plot.

![alt text][softmax_prob_2]

For the third image, the model is sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were shown in the below plot.

![alt text][softmax_prob_3]

For the fourth image, the model predicted that this is a Turn left ahead sign (probability of 1.0), although the image contained a 30Kmph road sign. This is probably caused due to the distorted image after resizing. The top five soft max probabilities were shown in the below plot.

![alt text][softmax_prob_4]

For the last image, the model is pretty sure that this is a Stop sign (probability of 0.9), and the image does contain a Stop sign. The top five soft max probabilities were shown in the below plot.

![alt text][softmax_prob_5]
