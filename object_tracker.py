import numpy as np
import cv2

from object_detection import ObjectDetection


od = ObjectDetection()
captured_vid = cv2.VideoCapture('los_angeles.mp4')
f_count = 0

# we will be breaking this loop during 2 conditions - when the ret is set to false, when the frame is None or when the user presses 'q' and it has been more than 1ms since the last key press.
while True:
    ret, frame = captured_vid.read()
    f_count += 1
    center_points = []
    if not ret or frame is None:
        break
    class_ids, scores, boxes = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box # the coordinates of the first point including the width and the height
        # we are trying to get the center coordinates of the box within a single frame
        center_x = int(x + x + w)//2
        center_y = int(y + y + h)//2
        center_points.append((center_x, center_y))
        for pt in center_points: # draws a circle around the center of all the obiects irrespective of the frame they are present in for purposes of object tracking
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        print(f"Box coordinates - frame count {f_count}:", x, y, w, h)
        # we will form a rectangle around the object using these coordinates - cv2.rectangle takes top left and bottom right coordinates
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 

    cv2.imshow('frame', frame)

    if cv2.waitKey(0) & 0xFF == ord('q'): # reads the last 8 bits of the key pressed and compares it with the ascii value of 'q' 
        break


#Releasing the captured video and closing the window
captured_vid.release()
cv2.destroyAllWindows()
