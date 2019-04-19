# Vehicle Detection Project

---

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train an SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/HOG_example.png
[image2]: ./exploration/000275.png
[image3]: ./output_images/000275_rgb.png
[image4]: ./output_images/000275_hls.png
[image5]: ./output_images/000275_hsv.png
[image6]: ./output_images/000275_luv.png
[image7]: ./output_images/000275_yuv.png
[image8]: ./output_images/000275_ycrcb.png
[image9]: ./output_images/spatial_bins.png
[image10]: ./output_images/normalized_features.png
[image11]: ./output_images/sliding_windows_raw.png
[image12]: ./output_images/sliding_windows.png
[image13]: ./output_images/bboxes_and_heatmaps.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Code pointers refer to sections in the [./Vehicle_Detection.ipynb](./Vehicle_Detection.ipynb) IPython notebook.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

The code for extracting HOG features from the training images is in Section 3 of the notebook and in the `extract_image_features()` function in the first cell of Section 4.

My overall strategy was to use spatial binning to capture color information and HOG features to capture shape information. In order to get good shape information, I wanted high contrast between the vehicle and its background. In the traffic signs project, I discovered that grayscaling is a good way to increase contrast, so I grayscale the images before calling `get_hog_features_one_channel()`, which is a wrapper around `skimage.hog()`, which actually computes the HOG features. Grayscaling reduces the shape information to a single channel, which has the additional benefit of reducing the number of HOG features that I need to compute. This speeds up both feature extraction and SVM training.

Here is an example of grayscaling a car image and computing its HOG features:

![alt_text][image1]

I experimented with parameters by training SVMs and looking at how test accuracy was affected. I started with 9 orientation bins, 8 pixels per cell, and 2 cells per block. I found that reducing the number of orientation bins (`ORIENT`) from 9 to 5 negatively impacted test accuracy, as did increasing the number of cells per block (`CELLS_PER_BLOCK`) from 2 to 3. However, I could maintain roughly the same test accuracy while substantially reducing the number of HOG features by increasing the number of pixels per cell (`PIX_PER_CELL`) from 8 to 16.

In the end, I chose not to increase the number of pixels per cell for two reasons:

1. by viewing HOG visualizations of random car images, I saw that increasing the number above 12 caused the HOG visualization to frequently not look much like a car anymore, and
2. a large pixel per cell value causes the sliding window search implementation using HOG sub-sampling to have very coarse-grained control over window tiling, which negatively impacts car detections "in the wild."

I augmented the HOG features with spatially-binned color features. This code for extracting the color features is in the last two cells of Section 2 of the notebook, and the `extract_image_features()` function in the first cell of Section 4.

I began by selecting a color space to work in, by taking example images and plotting a sample of their pixels in various color spaces.

Here is an example of a scene and a sample of its pipxels plotted in RGB, HLS, HSV, LUV, YUV, and YCrCb color spaces:

![alt_text][image2]
![alt_text][image3]
![alt_text][image4]
![alt_text][image5]
![alt_text][image6]
![alt_text][image7]
![alt_text][image8]

More examples can be found in Section 2 of [./output_images/Vehicle_Detection.html](./output_images/Vehicle_Detection.html).

I found that HLS color space maximized the distance between pixels associated with cars and background pixels. On close inspection, I found that the lightness (L) and saturation (S) channels did most of the separation, and hue (H) might vary, so I chose to only the L and S channels.

To determine the resolution of spatial bins to use, I resized input images to various sizes. I found that below about 10x10 resolution, it is hard to tell that the image is of a car. To divide the 64x64 input images evenly, I decided to use 16x16 bins.

Here is an example of what a car image looks like at this resolution:

![alt_text][image9]

Finally, I wrote a function `bin_spatial_ls()` that converts an image to HLS color space, extracts the L and S channels, and flattens them into a feature vector.

Finally, the `extract_image_features()` function extracts grayscaled HOG features, and L and S channel spatial binning features, and concatenates both into the final feature vector for training the SVM.

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training a classifier consisted of three steps:

1. Extracting and normalizing car and not-car data.
2. Splitting the data into training and test sets.
3. Using grid search to train SVM classifiers and select optimal hyperparameters.

The feature extraction and normalization code appears in Section 4 of the notebook.
The training and test set splitting and classifier training code appear in Section 5 of the notebook.

To extract features for all image, I used an `extract_features()` function that takes in a list of image paths and produces a list of feature vectors, one per image. This was done once for all car images and once for all not-car images. Both lists of feature vectors were then compiled into a matrix of feature vectors using `np.vstack()`. Similarly, labels of 1 and 0 were created for the car and not-car images, respectively, and concatenated using `np.hstack()` to produce a vector of output labels. The features were normalized to 0 mean and unit variance by creating a `sklearn.preprocessing.StandardScaler()`, using `fit()` to fit it to the feature matrix, and then applying `transform()` to the feature matrix.

Here is an example of unnormalized and normalized feature vectors for a random car image:

![alt_text][image10]

The feature matrix and labels vector were split into training and test sets by applying `sklearn.cross_validation.train_test_split()`.

Finally, I trained an SVM classifier using grid search to automate hyperparameter tuning. The process for this is to set up a dictionary of hyperparameter options (this is the `parameters` variable in the code). Then I create an SVM model using `sklearn.svm.SVC()` -- I set `probability=True` so that I can use the raw probability values to set a stricter prediction threshold during the sliding window search step later. Finally, I pass the SVM model and the parameters into `sklearn.grid_search.GridSearchCV()` and use the `fit()` function to begin model training using grid search.

The best parameter values found were to use an `rbf` kernel with a `C` value of 15. I did not provide `gamma` values in `parameter` because doing so caused grid search to take an egregiously long time to run. Test set accuracies were in the 98-99% range, varying depending on the run.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in the `find_cars()` function of Section 6 of the notebook.

My sliding window search implementation using HOG sub-sampling, to save time computing HOG features. The function works as follows:

1. HOG features are computed once for the entire search region of the image to produce the variable `hog`. The search region `ctrans_search` is defined by the constants `YSTART` and `YSTOP` and extend the entire width of the input image. The provided `scale` is also used to scale the image prior to computing HOG features. I made sure to grayscale the image prior to computing HOG features, the same way I did prior to model training. 
2. Then sliding windows are applied over the search region, stepping 1 cell right and/or down on each iteration. HOG features for each window are found by sub-selecting from the complete set of HOG features `hog` to produce the `hog_features` for the window under consideration. Color bin `spatial_features` (specifically, for just the L and S channels) are also computed for the window.
3. Using the fitted `X_scaler`, the features for the window are normalized and passed to a classifier `clf` to predict whether, with high probability, the patch of image within the window contains a car or not. The probability threshold is given by `PROB_THRESH`.
4. If a winow is predicted to contain a car (i.e. exceeds the threshold), that window is recorded.
5. Finally, the list of all windows predicted to contain cars is returned.

I used scales of 1, 2, and 3. Each scale requires computing a set of HOG features for the entire roadway search area, so I wanted to keep the number of scales small. At the same time, the scales need to be able to identify both nearby cars (that appear large) and cars further in the horizon (which appear small). I found that Scales of 3 and 1 captures these two cases, and I included one more scale of 2 to interpolate between the two.

HOG features are associated with a cell, so to overlap windows without needing to recompute HOG features, I need to step some integer number of cells each iteration. In my implementation, to try to maintain high granularity (i.e. low jitter), I use a step size of 1 cell.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

My final sliding window search uses grayscaled HOG features and spatially binned color features in the L and S channels. On each iteration, I step 1 HOG cell. The `YSTART` and `YSTOP` values used were 380 and 700.

To optimize the performance of my classifier, I computed the actual probability that a window contained a car using `clf.predict_proba()`, according to the model, then applied a very strict thresholding of `PROB_THRESH = 0.99`.

Without thresholding (i.e. using `clf.predict()` and checking that the result equalled 1), the test images would look something like this:

![alt text][image11]

With thresholding, false positives could be almost entirely eliminated without affecting true positives very much:

![alt text][image12]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To filter for false positives and combine ovelapping bounding boxes, I created a heatmap from the windows predicted to contain cars. I did this by creating an empty heapmap and applying the `add_heat()` function to apply heat from a list of bounding boxes. I then applied a threshold (`apply_threshold()` function) to zero out pixels without enough heat. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, which I assumed to correspond to a car. I then constructed bounding boxes (`draw_labeled_bboxes()` function) to cover the area of each blob detected. These steps are done in the `draw_bounding_boxes()` function.

Here is a copy of the test images after `find_cars()` was applied to them:

![alt text][image12]

This is what the test images look like after genereating heatmaps, applying a heatmap threshold value of `HEAT_THRESH = 1`, and using the result to draw bounding boxes around each labeled item:

![alt_text][image13]

For the most part, bounding boxes are tight, and there are no false positives. The black vehicle was not always detected. To increase the chance that it was detected during video generation, I collected all bounding boxes resulting from `find_cars()` from the last `FRAME_WINDOW = 20` frames of the video, then applied `HEAT_THRESH = 10` to eliminate false positives and tighten the final bounding boxes.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The biggest issue I faced when implementing this project was that training the SVM and generating the project video were extremely slow. In the end, it took almost 12 hours to generate the ouput project video! In a previous iteration, I had tried using 16 pixels per cell for HOG features (which reduces the number of features by about 4x), and the results were very jittery and contained too many false positives. The process could be improved by processing multiple segments of video in parallel and then stitching together the results.

* Another issue is that the HOG subsampling techniques makes defining the search window tiling less flexible. An alternative is to define a more general sliding window search, and compute HOG features from scratch for each window, but this comes at the cost of speed.

* Another issue was that completely eliminating false positives was difficult. The model tended to identify blocky-looking shapes on the other side of the highway divider (e.g. other cars, the railing, and the overhead sign/billboard) as cars. This might be reduced by applying another step prior to running the sliding window search, that masks out sections of image that are unlikely to be roadway. Smoothing out minor color striations in the roadway might also reduce the false positive identifications of cars directly in front of the camera. There was also a minor false negative detection that caused loss in tracking of the black car -- this might be improved by adding a shadow-removal component..