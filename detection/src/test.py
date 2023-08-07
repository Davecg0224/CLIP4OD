#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float64MultiArray, Int8
import numpy as np
import cv2

bridge = CvBridge()
pubImg = rospy.Publisher("webcam_img", CompressedImage, queue_size = 1)

rospy.init_node("webcam_node", anonymous = True)
rate = rospy.Rate(10)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
    
    imgmsg = bridge.cv2_to_compressed_imgmsg(img)
    pubImg.publish(imgmsg)
    rate.sleep()
    cv2.imshow("camera", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()