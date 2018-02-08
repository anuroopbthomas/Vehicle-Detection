**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[car_noncar]: ./my_images/car_noncar.png
[box1]: ./my_images/box1.png
[box125]: ./my_images/box125.png
[box15]: ./my_images/box15.png
[box18]: ./my_images/box18.png
[heatmap]: ./my_images/heatmap.png



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

Submission in project.ipynb



### Histogram of Oriented Gradients (HOG)

#### 1. Dataset exploration

I first explored the dataset of cars and non cars pictures. This dataset comes from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip and https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip .

![alt text][car_noncar]

There are 8792 car pictures and 8968 non-car pictures. These pictures will be used to train the classifier.


#### 1. Classification algorithm

As shown in the class lessons, I used a combination of
- HOG features (function `get_hog_features`)
- spatial features (function `bin_spatial`)
- color histograms (function `color_hist`)

to train a linear SVM classifier.

These three functions are combined in the `extract_features` function.

#### 2. Choice of parameters.
```python
color_space = "LUV" # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

orient = 11  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

spatial_size = (16,16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
```

The `hog_channel` was set to "ALL" because using a single channel proved to not be sufficient for proper results.



#### 3. Classifier training

I trained a linear SVM using 80% of the whole dataset of cars and non cars with the HOG+spatial+color features as described above. The remaining 20% are used as a validation set for computing the accuracy.

I was able to get an accuracy of 0.9924 (99.24%) on a validation set.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the implementation of the sliding window function of the lesson 35 and refactored it a bit so that it does not return a picture but only the identified bounding boxes.

I added a `confidence_threshold` parameter which filters the candidates and mandates that the classification 'confidence' is high enough (see line `if test_prediction == 1 and conf > confidence_threshold`).


#### 2. Result on a test image

Results...

- scale 1
![alt text][box1]
- scale 1.25
![alt text][box125]
- scale 1.5
![alt text][box15]
- scale 1.8
![alt text][box18]


We see that various scales compute different bounding boxes. We'll use a combination of these scales in the full pipeline.


### Heat maps

Heat map on an image

![alt text][heatmap]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For the full processing pipeline, I designed the `CarTracker` Python class. It combines multiple bounding boxes that are the results of the `find_cars` function:
- multiple scales will be used on each frame
- the most recent frames will be taken into account

Once we have all these bounding boxes, a heat map is made and a minimum threshold is applied.

We will maintain this list of recent and multiple-scale bounding boxes in a finite deque data structure. Its capacity is set to (number of recent frames to consider) X (number of scales). (see https://docs.python.org/3/library/collections.html#collections.deque)

This class is instantiated with:
```
- scales = (1, 1.25, 1.5, 1.8)
- cache = 16
- heat_threshold = 9
- confidence_threshold = -1.5
```
These parameters were chosen after a trial and error process in which I tried to minimize the number of wrong bounding boxes (false positive and false negatives) on the project video.

---

### Discussion

This project was quite straight forward. A few remarks though:
- many parameters were set after trial and errors. In particular the final choice of `cache=16, heat_threshold=9, confidence_threshold=-1.5` for the pipeline was done afet many trials to minimize the level of false positives and false negatives around second 23 - second 28 on the project video.
- long processing : processing the complete video required 45+ minutes on my laptop. Using parallel computation, for example with the Python `multiprocessing` module, could mitigate this issue, but the fundamental problem is that HOG+SVM is indeed very slow, and not really suited to real-time processing.
- a more modern approach using CNN object detection would certainly be better and more efficient. I plan on trying using the YOLO algorithm in the following weeks.
