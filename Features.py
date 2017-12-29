from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import pickle


with open('matrix.pickle', 'rb') as handle:
    mtx = pickle.load(handle)

with open('coeff.pickle', 'rb') as handle:
    dist = pickle.load(handle)


# Function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Function to extract features from a list of images. It calls bin_spatial(), color_hist() and get_hog_features()
def extract_features(imgs, cspace = 'RGB', spatial_size = (32, 32),
                        hist_bins=32, hist_range = (0, 256), orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features,hog_features)))
    # Return list of feature vectors
    return features


def extract_features_single_image(image, cspace = 'RGB', spatial_size = (32, 32),
                        hist_bins=32, hist_range = (0, 256), orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
    # if hog_channel == 3:
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    # Append the new feature vector to the features list
    features.append(np.concatenate((spatial_features, hist_features,hog_features)))
    # Return list of feature vectors
    return features


# Function to draw boxes in an image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Function that takes an image, start and stop positions in both x and y,  window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
# def search_windows(img, windows, clf, scaler, color_space='RGB',
#                     spatial_size=(32, 32), hist_bins=32,
#                     hist_range=(0, 256), orient=9,
#                     pix_per_cell=8, cell_per_block=2,
#                     hog_channel=0, spatial_feat=True,
#                     hist_feat=True, hog_feat=True):
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        # features = single_img_features(test_img, color_space=color_space,
        #                     spatial_size=spatial_size, hist_bins=hist_bins,
        #                     orient=orient, pix_per_cell=pix_per_cell,
        #                     cell_per_block=cell_per_block,
        #                     hog_channel=hog_channel, spatial_feat=spatial_feat,
        #                     hist_feat=hist_feat, hog_feat=hog_feat)
        features= extract_features_single_image(image = test_img, cspace=color_space,
                                                spatial_size=spatial_size, hist_bins=hist_bins,
                                                orient=orient, pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                hog_channel=hog_channel)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    # fig , axarr = plt.subplots(3, 2) # for report, add the images and change vis flag to use it
    # axarr[0, 0].imshow(ch1)
    # axarr[0, 0].set_title('CH1')
    # axarr[0, 1].imshow(hog1img)
    # axarr[0, 1].set_title('Hog 1')
    # axarr[1, 0].imshow(ch2)
    # axarr[1, 0].set_title('CH2')
    # axarr[1, 1].imshow(hog2img)
    # axarr[1, 1].set_title('Hog 2')
    # axarr[2, 0].imshow(ch3)
    # axarr[2, 0].set_title('CH3')
    # axarr[2, 1].imshow(hog3img)
    # axarr[2, 1].set_title('Hog 3')
    # fig.tight_layout()
    # plt.show()

    boxes =[]

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 255, 0), 6)
                boxes.append(((xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return draw_img, boxes



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

box_list_10 = []

def pipeline(image,scale):
    global box_list_10
    # input_image = mpimg.imread(image)
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    input_image = undistorted.astype(np.float32)/255

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # out_img,box_list = find_cars(input_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)
    for factor in range(1,scale):
        # _, box_list = find_cars(input_image, ystart+(50*(scale-factor)), ystop-(50*(factor-1)), (scale)/factor, svc, X_scaler, orient, pix_per_cell,
        #                               cell_per_block, spatial_size,
        #                               hist_bins)
        _, box_list = find_cars(input_image, ystart , ystop - (50 * (factor-1 )),
                                (scale) / ((factor)), svc, X_scaler, orient, pix_per_cell,
                                cell_per_block, spatial_size,
                                hist_bins)
        box_list_10.append(box_list)



    # Add heat to each box in box list
    for box_list_i in box_list_10:
        heat = add_heat(heat, box_list_i)


    # Remove elements with negative index equal to maximum number of iterations (n) until the end of the array [-n:_]
    if (len(box_list_10) > 10):
        box_list_10 = box_list_10[-10:]

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 6)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(undistorted), labels)
    # Plots for the report
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # plt.show()
    #
    # plt.imshow(labels[0], cmap='gray')
    # plt.show()
    # return draw_img,box_list
    return draw_img


def process_image(image):

    result=pipeline(image,5) #Number of scales to be used, it must be max 5

    return result

# Read in car and non-car images
images_cars = glob.glob('training_images_cars/*.png')
images_notcars= glob.glob('training_images_notcars/*.png')
cars = []
notcars = []
for image in images_cars:
    cars.append(image)

for image in images_notcars:
    notcars.append(image)

# Uncomment for report images
# carimg =mpimg.imread(cars[50])
# notcarimg = mpimg.imread(notcars[50])
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(carimg)
# plt.title('Car')
# plt.subplot(122)
# plt.imshow(notcarimg)
# plt.title('Not Car')
# fig.tight_layout()
# plt.show()
# print(len(notcars))
# print(len(cars))

# Experiment with these values to see how your classifier
# performs under different binning scenarios
spatial = 64
spatial_size = (spatial, spatial)
histbin = 64
hist_bins = histbin
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8 #16
cell_per_block = 2 #8
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
y_start_stop = [400, 650] # Min and max in y to search in slide_window()
ystart = 400
ystop = 650
#scale = 4


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

project_video1 = 'project_video_out_1.mp4'
clip1 = VideoFileClip('project_video.mp4')
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_video1, audio=False)

# test_images = glob.glob('test_images/*.jpg')
#
# for test_image in test_images:
#
#     image = mpimg.imread(test_image)
#     intermediate = cv2.undistort(image, mtx, dist, None, mtx)
#     out_img,list=process_image(image)
#
#     windows=draw_boxes(intermediate, list)
#     plt.imshow(windows)
#     plt.show()
#
#     # image = cv2.undistort(image, mtx, dist, None, mtx)
#     # draw_image = np.copy(image)
#     #
#     # # Uncomment the following line if you extracted training
#     # # data from .png images (scaled 0 to 1 by mpimg) and the
#     # # image you are searching is a .jpg (scaled 0 to 255)
#     # image = image.astype(np.float32)/255
#     #
#     # heat = np.zeros_like(image[:, :, 0]).astype(np.float)
#     # # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
#     # #                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
#     # #
#     # # hot_windows = search_windows(image, windows, svc, X_scaler, color_space=colorspace,
#     # #                         spatial_size=(spatial,spatial), hist_bins=histbin,
#     # #                         orient=orient, pix_per_cell=pix_per_cell,
#     # #                         cell_per_block=cell_per_block,
#     # #                         hog_channel=hog_channel)
#     # #
#     # # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
#     # #
#     # # plt.imshow(window_img)
#     # out_img,box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
#     #                     hist_bins)
#
#     plt.imshow(out_img)
#     plt.show()
#     # Add heat to each box in box list
#     # heat = add_heat(heat, box_list)
#     #
#     # # Apply threshold to help remove false positives
#     # heat = apply_threshold(heat, 1)
#     #
#     # # Visualize the heatmap when displaying
#     # heatmap = np.clip(heat, 0, 255)
#     #
#     # # Find final boxes from heatmap using label function
#     # labels = label(heatmap)
#     # draw_img = draw_labeled_bboxes(np.copy(image), labels)
#     #
#     # fig = plt.figure()
#     # plt.subplot(121)
#     # plt.imshow(draw_img)
#     # plt.title('Car Positions')
#     # plt.subplot(122)
#     # plt.imshow(heatmap, cmap='hot')
#     # plt.title('Heat Map')
#     # fig.tight_layout()
#     # plt.show()
