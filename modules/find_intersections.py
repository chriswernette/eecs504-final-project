#find_intersections.py
#finds intersections of Hough lines from input lines, returns array of (x,y)
#coordinates of intersection

import numpy as np
DEBUG = False

def find_intersection(line1, line2):
    '''function to find the intersections of 2 Hough lines. The results are
    returned in a #intersections X 2 numpy array, with each row being (x,y)'''

    '''since we are given two points on the lines, which are endpoints we can 
    use something called the first degree Bezier parameters solved by determinants
    In short, L1 = (x1) + t*(x2-x1)  L2 = (x3) + u*(x4-x3)
                   (y1)     (y2-y1)       (y3) + u*(y4-y3)'''
    #
    

    '''technically the segment is parameterized for 0 <= u,t <= 1 by definition,
    but it is a good idea to relax that a little to check for lines that are 
    almost touching, or points that are intersecting at the very ends'''
    #print(line1)
    #print(line2)

    #initialize some variables, to prevent errors
    intersection = []
    checkT = 0
    checkU = 0
    xInt = 0
    yInt = 0
    
    #grab coords from 2 lines
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2

    #compute angles and difference, which we will use later
    #theta1 = np.arctan2(x2-x1,y2-y1) * 180/np.pi
    #theta2 = np.arctan2(x4-x3,y4-y3) * 180/np.pi
    theta1 = np.arctan2(y2-y1,x2-x1) * 180/np.pi
    theta2 = np.arctan2(y4-y3,x4-x3) * 180/np.pi
    angleDiff = abs(theta1-theta2)

    #pre-compute denominator so we don't accidentally divide by 0
    denominator = ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))

    if(abs(denominator) > .001):
        #formula from wikipedia for finding parameterization of line
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/denominator
        u = -((x1-x2)*(y1-y3) -(y1-y2)*(x1-x3))/denominator

        #can give a little wiggle room for rounding error if needed
        checkT = (0.0<=t) and (t<=1.0)
        checkU = (0.0<=u) and (u<=1.0)
        #print('The value of t is ' + str(t))
        #print('The value of u is ' + str(u))

        #set the xInt and yInt values, also from wikipedia
        if(checkT):
            xInt = x1 + t*(x2-x1)
            yInt =  y1 + t*(y2-y1)
        if(checkU):
            xInt = x3 + u*(x4-x3)
            yInt = y3 + u*(y4-y3)
        
        #add to intersection list
        if((checkT or checkU) and ((angleDiff > 75) and (angleDiff < 105))):
            if DEBUG:
                print('This intercept is (' + str(xInt) + ',' + str(yInt) + ')')
                print('and the angle difference is')
                print(angleDiff)
            intersection = [xInt,yInt]

    return intersection 


def segmented_intersections(lines):
    #function to find the intersection between all lines

    '''got this from online. Makes it so you only grab unique intersections,
    I don't understand it that well tbh'''
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(find_intersection(line1, line2)) 
    
    #remove empty items in the list
    intersections = [t for t in intersections if t]
    intersections = np.asarray(intersections)
    return intersections
