# Billboard Detection and Projection onto Car Windshield with Eye Tracking
This is our final project for Fall '18 EECS 504 - computer vision at the University of Michigan. Our group consists of Alex Crean, Moe Farhat, Peter Paquet, and Chris Wernette. Our project is a proof of concept of a product that can detecting billboards while driving down the road from a dash camera, and then projecting those billboards onto the windshield at the passenger's current point of view.

# Motivation
Billboards on the side of highways are a common type of ad in the United States. However, the driver must take their eyes off the road to look at the ad, which is both unsafe, and therefore leads to lower time looking at the ads. In the not too distant future, we envision autonomous vehicle systems allowing the driver to not have to pay attention to the road. In these scenarios, marketing agencies will want to increase the amount of time the passengers are looking at ads. In order to do this, we can use a dash camera to detect billboards in the current frame, and then project that billboard onto the windshield at the passenger's current point of view.

# Screenshots/Demo Video
Four sample gifs of raw input data and the resulting frame-by-frame windshield projections are found in the `gif_demos/` folder.

# Code Example 1
To run the billboard-detector_mult_crop_window algorithm, use the following command: 
`python3 billboard-detector_mult_crop_window.py`
There are a few variables in the `main()` function that can be changed. To run the algorithm over an entire dataset, set `mode = 1` (line 120) and set `path` (line 130) to the folder corresponding to the dataset of choice. In mode 1, the windshield projection images will be saved in the `output/` directory, with file names corresponding to the input image frame used to make the projection. If the `output/` directory does not already exist, please initialize it.
To run the algorithm on a single image, set `mode = 0` and set `files` (line 125) to the image of choice.
For more verbose output and to see the intermediate steps of Canny edge detection, Hough transform, Hough intersections, and billboard masking, set `DEBUG = True` on line 32.
As the algorithm is run frame by frame projections or no billboard found is displayed, and then the output directory incorporates use of adlock to project to frames that are "locked".

# Code Example 2
To run the billboard-detector cropped images algorithm, use the following command:
`python3 billboard-detector.py`
The mode variable in the `main()` function can be changed to run the whole dataset or just one image. See directions above. You can also change the tune from loose to tight at the top of the file. We prefer tight, which needs to be closer to find the billboard but has more accurate results. 
There is also the option to run the algorithm in DEBUG mode, which shows each of the steps of our algorithm, frame by frame.
The output of the algorithm frame by frame will be saved in the /output folder which is automatically created and deleted each time the script is run. 
As the algorithm is run frame by frame projections or no billboard found is displayed, and then the output directory incorporates use of adlock to project to frames that are "locked".

# Installation
`git clone https://github.com/chriswernette/eecs504-final-project.git`
This will download all the files necessary for the project.

# Modules

## billboard-detector.py
File that will set up input files, call other modules. I intend to set this up so you can specify high level args like video filename/images directory when you call it from the command line.

## preprocessing.py
This function takes in an image and handles all the preprocessing. The output will be Hough lines in (x1, y1), (x2, y2) format. Submodules of this function are cropping, edge detection, and Hough lines. The preprocessing module will make calls to the sign masking module to exclude edges/Hough lines resulting from Michigan highway signs.

## street_sign_mask.py
This will take in an image as input and return an image mask of typical traffic signs like exit information, contruction signs, and road information. Traffic signs are detected based on the official color values for green, yellow, orange, and blue traffic signs. The output is used by preprocessing.py to exclude these types of rectangular objects on the road.

## find_intersections.py
This will take in the endpoints of all the lines, and then find their intersections by parameterizing each line. Also, it will reject intersections that are not close to 90 degrees to eliminate false positives. The output of this function is a list of (x,y) coordinates that meet the criteria of being a valid intersection of two nearly perpendicular lines. See [Line-Line Intersection](https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection) given two points on each line section.

## cluster_corners.py
This function will take in the list of intersections, and bin nearby intersections to a centroid. The output of this is a reduced number of possible intersections for the corners of the billboard. Uses the Mean Shift clustering method from scikit-learn, you can find a demo here [A demo of the mean-shift clustering algorithm](https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html) and a summary of clustering algorithms available from scikit-learn can be found here [2.3 Clustering](https://scikit-learn.org/stable/modules/clustering.html#mean-shift)

## Billboard Corner Hypothesis
This function takes, as input, the set of Hough lines intsection centroids. A convex hull is fit to these clusters, which is combined with the input image to yield the billboard mask. The clusters are then sorted in way to approximate the four corners of a bounding box for the hypothetical billboard. The billboard mask and bounding box corners are outputs of the function.

## Billboard Rejection
This function takes, as input, the set of corner outputs from the Billboard Corner Hypothesis function. The rejection function then calculates the angles and side lengths of the hypothetical bounding box. If the angles or sides are outside of the specified tolerance, the function rejects the billboard hypothesis.

## Billboard extraction and projection onto Vehicle Windshield
This function takes the 4 corner hypothesis and projects the billboard onto the Heads Up Display(HUD). The projection is a function of where the passengers eyes are fixed onto the windshield. 
