## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/figure1.png
[image2]: ./output_images/figure2.png
[image3]: ./output_images/boxes2.png
[image4]: ./output_images/heatmap.png
[image5]: ./output_images/heatmap2.png
[image6]: ./output_images/labeled.png
[image7]: ./output_images/labeled2.png
[image8]: ./output_images/heatmap4.png
[image9]: ./output_images/heatmap41.png
[video1]: ./project_video_out_1.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 58 through 99 of the file called `Features.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images and storing them in corresponding lists:

```python
# Read in car and non-car images
images_cars = glob.glob('training_images_cars/*.png')
images_notcars= glob.glob('training_images_notcars/*.png')
cars = []
notcars = []
for image in images_cars:
    cars.append(image)

for image in images_notcars:
    notcars.append(image)
```
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). In order to give an intuitive understanding of this, here is an example using the `YCrCb` color space and HOG parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, applied to the trimmed region of interest in the image (i.e. the area where vehicle images are expected to appear):

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters trying to reduce the time needed to process and render the video and using the combination that produced the best results in vehicle detection. The final values for the parameters are: `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Reducing the number of pixels per cell proved to increase the number of false positives signifficantly.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM as shown in lines 484 through 529 of the file called `Features.py`. After data was split in training and test sets YCrCb 3-channel HOG features plus spatially binned color and histograms were stacked were extracted from training images and stacket in a vector to feed the SVM trainer.

```python
car_features = extract_features(cars, cspace = colorspace, spatial_size = (spatial, spatial),
                        hist_bins = histbin, hist_range = (0, 256), orient  = orient,
                        pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,
                        hog_channel = hog_channel)
notcar_features = extract_features(notcars, cspace = colorspace, spatial_size = (spatial, spatial),
                        hist_bins = histbin, hist_range  = (0, 256), orient = orient,
                        pix_per_cell  = pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel = hog_channel)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
```

Results of running this code section are ouput in the console:

```sh
Using spatial binning of: 64 and 64 histogram bins
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 17772
98.15 Seconds to train SVC...
Test Accuracy of SVC =  0.9913
My SVC predicts:  [ 0.  0.  1.  1.  1.  0.  0.  0.  0.  1.]
For these 10 labels:  [ 0.  0.  1.  1.  1.  0.  0.  0.  0.  1.]
0.0161 Seconds to predict 10 labels with SVC
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at a number of scales defined as a parameter (in order evaluate the combination of several scales), limited to positions in Y going from 400 to 650 pixels in the image. This can be found in the function `find_cars()` in lines 254 through 338 of the file called `Features.py`).  


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales, limiting the area to be considered per scale and using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  Here are some example images:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out_1.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions and later on I used the function`scipy.ndimage.measurements.label()` to identify individual elements visible in the heatmap, which were assumed to be a vehicle. I accumulated the heatmap measurements over 10 frames and then I built bounding boxes to cover the area of each detected object (pixel count above a certain threshold).  

This is implemented in function `pipeline()`shown in lines 377 through 432 of the file called `Features.py`

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are a couple of frames with the output boxes and the corresponding heatmaps used to build them:

![alt text][image4]

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the shown frames:

![alt text][image6]

![alt text][image7]

### Here is an example of the detected boxes used as input to the heatmap and the resulting bounding boxes drawn onto the frame:

![alt text][image8]

![alt text][image9]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline has shown weakness in areas with guard rails and trees when the image resembles somewhat some of the visual features of a car. Further investigation in the combination of hog features and color spaces might be benefitial for improving detection performance.

The filtering of false positives comes with a trade-off with the system reactivity. We must strive for an optimal balance between a fast (reactive) detection system and an accurate one.

An interesting way to render more robust the pipeline would be to use parallel redundant classifiers, e.g. a CNN and a SVM trained on different sets (and using complementary features or color spaces) of contextually similar data and initializing detected object only when both classifiers agree upon on the existance of a given object.



