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
    if not ret or frame is None:
        break
    class_ids, scores, boxes = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box # the coordinates of the first point including the width and the height
        print(f"Box coordinates - frame count {f_count}:", x, y, w, h)
        # we will form a rectangle around the object using these coordinates - cv2.rectangle takes top left and bottom right coordinates
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # reads the last 8 bits of the key pressed and compares it with the ascii value of 'q' 
        break


#Releasing the captured video and closing the window
captured_vid.release()
cv2.destroyAllWindows()
