import cv2
import numpy as np

#Based on https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
  
def frame_to_jpeg(video_path, t_start, t_end):
    #Create Video Capture
    vid_cap = cv2.VideoCapture('2018_1130_120220_027.MOV')
    count = 0
    frame_count = 0
    # Determine Start and End Frame - 30fps
    start_frame = t_start*30
    end_frame = t_end*30

    # Check if camera opened successfully
    if (vid_cap.isOpened()== False): 
      print("Video Not Loaded")
    # Read Video
    while(vid_cap.isOpened()):
      # Read Frame
      success, frame = vid_cap.read()
      # Frame is read successfully
      if success == True:
        count += 1
        # While current frame is within the start and end frame bounds, save frame as .jpg
        if (end_frame >= count >= start_frame):
            #Saves a frame every half second (15 frames)
            if (count % 15 == 0):
                #cv2.imshow('Frame',frame)
                frame_count += 1
                cv2.imwrite("frame%d.jpg" % frame_count, frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

        if count == end_frame+1:
            break
      # Break the loop
      else: 
        break

    # When everything done, release the video capture object
    vid_cap.release()
    return

#frame_to_jpeg('video_path',t_start, t_end)