#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import math

depth_buffer = []
depth_msg = None
camera_info_buffer = []
camera_info_msg = None
camera_msg = None
msg_list = []
depth = []
cmd = Twist()
yaw = 0
tmp = 0
count = 0
count2 = 0
inf = float("inf")
speed = 1

def camera_callback(msg):
    global camera_msg, msg_list
    camera_msg = msg
    if depth_msg is None or camera_info_msg is None or camera_msg is None:
        return
    msg_list.append([camera_msg, copy.deepcopy(depth_msg), copy.deepcopy(camera_info_msg)])

def depth_callback(msg):
    global depth_msg, depth_buffer, depth
    depth = []
    depth_msg = msg
    depth_buffer.append(msg)
    
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(depth_msg)
    # rospy.loginfo(depth_image.shape)
    # shape = depth_image.shape[]
    # for i in range(5):
    # rospy.loginfo("depth_msg")
    # rospy.loginfo(np.nanmean(depth_image[:, 0:256]))
    
    depth.append(np.nanmean(depth_image[360:, 0:256]))
    # rospy.loginfo(np.nanmean(depth_image[:, 0:256]))
    depth.append(np.nanmean(depth_image[360:, 256:512]))
    # rospy.loginfo(depth)
    depth.append(np.nanmean(depth_image[360:, 512:768]))
    # rospy.loginfo(depth)
    depth.append(np.nanmean(depth_image[360:, 768:1024]))
    # rospy.loginfo(depth)
    depth.append(np.nanmean(depth_image[360:, 1024:1280]))
    rospy.loginfo(depth)
    # cv2.imshow("show",depth_image)
    # cv2.waitKey(1)
    # cv2.destroyWindow("show")
    
    rotate_degree()
    rospy.sleep(2)

    move_straight_time("forward", speed, depth[tmp]/speed)#0.5 change to depth[tmp]        
    rospy.sleep(2)
    # rospy.loginfo("depth_buffer")
    # rospy.loginfo(depth_buffer)   

def camera_info_callback(msg):
    global camera_info_msg, camera_info_buffer
    camera_info_msg = msg
    camera_info_buffer.append(msg)
def move_straight_time(motion, speed, time):

    global count2
    rospy.loginfo("move forward")
    count2 = 0
    # Initilize velocities
    cmd.linear.y = 0
    cmd.linear.z = 0
    cmd.angular.x = 0
    cmd.angular.y = 0
    cmd.angular.z = 0

    if motion == "forward":
        cmd.linear.x = speed
        if cmd.linear.x == inf:
            cmd.linear.x = speed
    elif motion == "backward":
        cmd.linear.x = - speed

    i = 0
    # loop to publish the velocity estimate, current_distance = velocity * (t1 - t0)
    while (1):
        print("=======Now moving========")
        print("time:",round(time, 0)*10)
        # print (cmd.linear.x)
        # Publish the velocity
        pub_vel.publish(cmd)
        # i += 0.1
        count2 += 1 
        rospy.sleep(0.1)
        if count2 == round(time,0)*10:
            print("cmd:", cmd)
            # print("time:", count2)
            break
        
    cmd.linear.x = 0
    # set velocity to zero to stop the robot
    # stop_robot()

def rotate(degrees):
    global count
    count = 0
    rospy.loginfo("rotate")
    # time.sleep(1)
    target_rad = (degrees * math.pi/180)

    # if target_rad < (- math.pi):
    #     target_rad = target_rad + (2 * math.pi)

    # if target_rad > (math.pi):
    #     target_rad = target_rad - (2 * math.pi)

    # while abs(target_rad - yaw) > 0.01:
    #     print("hahahahhaha")
    #     cmd.angular.z = 0.5 * (target_rad - yaw)
    #     pub_vel.publish(cmd)
    #     rate.sleep()
    print(depth[tmp])
    if (depth[tmp]<1.5):
        print("mode2")
        while (1):
            target_rad = (-44 * math.pi/180)
            # cmd.linear.x = 1
            cmd.angular.z = 10 * (target_rad - yaw)
            pub_vel.publish(cmd)
            print(cmd.linear.x)
            print(degrees)
            # print("target = {} current:{}",degrees, yaw)
            count += 1
            rospy.sleep(0.1)
            if count == 20:
                break
    else :
        print("mode1")
        while (1):
            # cmd.linear.x = 1
            cmd.angular.z = 2.5 * (target_rad - yaw)
            pub_vel.publish(cmd)
            print(cmd.linear.x)
            print(degrees)
            # print("target = {} current:{}",degrees, yaw)
            count += 1
            rospy.sleep(0.1)
            if count == 10:
                break

    # print ("hahahahah")

    cmd.angular.z = 0
    # stop_robot()

def stop_robot():
        rospy.loginfo("shutdown time! Stop the robot")
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        pub_vel.publish(cmd)    

def odom_callback(msg):
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

def rotate_degree():
    global depth,tmp 
    tmp = 2
    longest = 0
    for i in range(len(depth)):
        if depth[i]>longest:
            longest = depth[i]
            tmp = i 
    print("tmp",tmp)
    if tmp == 0:
        rotate(44) 
    if tmp == 1:
        rotate(22) 
    if tmp == 2: 
        rotate(0)
    if tmp == 3:
        rotate(-22) 
    if tmp == 4:
        rotate(-44)

        


if __name__ == '__main__':
    rospy.init_node("test_node", anonymous = True)
    pub_vel = rospy.Publisher('/temp_cmd_vel', Twist, queue_size = 10)
    # rgb_image
    # rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color/compressed", CompressedImage, callback=camera_callback, queue_size=1)
    # depth_image
    # rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, callback=depth_callback, queue_size=1)
    # camera info
    # rospy.Subscriber("/zed2/zed_node/rgb/camera_info", CameraInfo , callback=camera_info_callback, queue_size=1)
    # rospy.loginfo()
    # while not rospy.is_shutdown():
    rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, callback=depth_callback, queue_size=1)
    rospy.Subscriber ('/odom', Odometry, odom_callback)#要去查一下要訂閱哪個node可以得到現在的角度

    rate = rospy.Rate(10)
    rospy.spin()
        