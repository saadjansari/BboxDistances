import numpy as np
import cv2

def read_video(vid_path):
    imgs = []

    # Initialize Video
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(vid_path)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Change RGB for frame
            imgs.append( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return imgs


def save_video(img_array, vid_path,fps=10):
    out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'MP4V'), 
                        fps, (img_array[0].shape[1],img_array[0].shape[0]))
    for i in range(len(img_array)):
        out.write(cv2.cvtColor(img_array[i], cv2.COLOR_BGR2RGB))
    out.release()
