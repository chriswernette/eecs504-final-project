# Billboard Detection and Projection onto Car Windshield with Eye Tracking
This is our final project for Fall '18 EECS 504 - computer vision at the University of Michigan. Our group consists of Alex Crean, Moe Farhat, Peter Paquet, and Chris Wernette. Our project is a proof of concept of a product that can detecting billboards while driving down the road from a dash camera, and then projecting those billboards onto the windshield at the passenger's current point of view.

# Motivation
Billboards on the side of highways are a common type of ad in the United States. However, the driver must take their eyes off the road to look at the ad, which is both unsafe, and therefore leads to lower time looking at the ads. In the not too distant future, we envision autonomous vehicle systems allowing the driver to not have to pay attention to the road. In these scenarios, marketing agencies will want to increase the amount of time the passengers are looking at ads. In order to do this, we can use a dash camera to detect billboards in the current frame, and then project that billboard onto the windshield at the passenger's current point of view.

# Screenshots/Demo Video
Add a youtube link here later or maybe a some screenshots of intermediate steps like Hough lines, detecting intersections of lines, etc.

# Code Example
Add a few lines of code that quickly allow the user to run the project, then add a few that would show them how to tweak parameters.

# Installation
`git clone https://github.com/chriswernette/eecs504-final-project.git`
This will download all the files necessary for the project. To run the project off some of our example videos simply type:
@TODO Define this later
`python3 wrapper-function-name.py input-video.filetype`

# Modules
This section has a basic list of who is doing what and describes what the inputs and outputs of the modules should look like.

## Wrapper function - team
File that will set up input files, call other modules. Basically a main function that will call all the modules we are writing one after another.

## Cropping - Chris
This function takes in an image and crops it to be in the quadrant specified in the main wrapper function.

## Edge Detection - Chris
This will take in the cropped image and output a binary image using Canny edge detection or Harris Corners.

## Sign Masking - Alex
@TODO Alex describe however you want.

## Hough Lines -Chris
This will take in the binary edge detected image and the mask to ignore Michigan highway signs, and remove edges detected within those masks, and then on the remaining edges it will apply the Hough lines detection algorithm. The output of this function will be a set of 2 points which can be used to define a line. At each element of the last are (x1,y1) (x2,y2).

## Hough Lines Intersections - Chris
This will take in the endpoints of all the lines, and then find their intersections by parameterizing each line. Also, it will reject intersections that are not close to 90 degrees to eliminate false positives. The output of this function is a list of (x,y) coordinates that meet the criteria of being a valid intersection of two nearly perpendicular lines.

## Hough Lines Intersections Clustering - Chris
This function will take in the list of intersections, and bin nearby intersections to a mean intersection. It may also weight them based on how many intersections that are nearby and how close they are to each other. The output of this should be a reduced number of possible intersections for the corners of the billboard.

## Billboard Corner Hypothesis - Peter
@TODO Peter describe however you want.

## Homography - Moe
@TODO Moe describe however you want

## Stitch Homography Images into Video - Team
@TODO I think if we put the adds projected onto the windshield all in one array this should be super easy there's gotta be code to do this.
