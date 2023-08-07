# -*- coding: UTF-8 -*-
#! /usr/bin/env python
import rospy
from std_msgs.msg import String, Float64MultiArray, Int8
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from cv_bridge import CvBridge
import numpy as np
import time
import math

received_image = np.array([0,0,0,0,0,0,0])
old_received_image = []
target_floor = 0
target_room = 0
cmd = Twist()
ctrl_c = False
roll = 0
pitch = 0
yaw = 0
door_num = np.array([0,0])
door_num = Float64MultiArray(data = door_num)   
keepGoing = 0
ISee = 0
IHeard = 0
cur_door_num = 0
movemode = 0
waitForEleOpen = 0
mention = 0
depth_buffer = []
depth_msg = None
camera_info_buffer = []
camera_info_msg = None
camera_msg = None
msg_list = []
depth = []
tmp = 0
count = 0
count2 = 0
inf = float("inf")

def targetFloor(msg):
    global target_floor
    global target_room
    target_floor = int(msg.data[0])
    target_room = int(msg.data[1])
    # rospy.loginfo("target floor")
    # rospy.loginfo(target_floor)
    # rospy.loginfo("target room")
    # rospy.loginfo(target_room)

def ImageDepth(msg):
    for i in range(len(msg.data)):
        received_image[i] = msg.data[i]
    # rospy.loginfo(received_image)    
        
def SeeSomething(msg):
    global ISee
    ISee = msg.data
    # rospy.loginfo(ISee)    

def SeeDoorNum(msg):
    global cur_door_num
    cur_door_num = msg.data
    door_num.data[1] = cur_door_num
    
    rospy.loginfo(door_num.data[0])
    rospy.loginfo(door_num.data[1])  

def HeardSomething(msg):
    global IHeard
    IHeard = msg.data
    # rospy.loginfo(IHeard)

def isKeepGoing(msg):
    global keepGoing
    keepGoing = msg.data
    # rospy.loginfo(keepGoing)

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
    depth.append(np.nanmean(depth_image[360:, 256:512]))
    depth.append(np.nanmean(depth_image[360:, 512:768]))
    depth.append(np.nanmean(depth_image[360:, 768:1024]))
    depth.append(np.nanmean(depth_image[360:, 1024:1280]))
    rospy.loginfo(depth)
    
    rotate_degree()
    rospy.sleep(2)
    move_straight_time("forward", depth[tmp]*1.5, 1)#0.5 change to depth[tmp]

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
            cmd.linear.x = 1.5
    elif motion == "backward":
        cmd.linear.x = - speed

    i = 0
    # loop to publish the velocity estimate, current_distance = velocity * (t1 - t0)
    while (1):
        print (cmd.linear.x)
        # Publish the velocity
        pub_vel.publish(cmd)
        # i += 0.1
        count2 += 1 
        rospy.sleep(0.1)
        if count2 == 20:
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

    print ("hahahahah")

    cmd.angular.z = 0
    # stop_robot()

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

def stop_robot():
        rospy.loginfo("shutdown time! Stop the robot")
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        pub_vel.publish(cmd)
        
#         publish_once_in_cmd_vel()

# def publish_once_in_cmd_vel():
#     """
#     This is because publishing in topics sometimes fails the first time you publish.
#     In continuos publishing systems there is no big deal but in systems that publish only
#     once it IS very important.
#     """
#     while not ctrl_c:
#         connections = pub_vel.get_num_connections()
#         if connections > 0:
#             pub_vel.publish(cmd)
#             rospy.loginfo("Cmd Published")
#             break
#         else:
#             rospy.loginfo("i sleep")
#             rate.sleep()

def odom_callback(msg):
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)



if __name__ == '__main__':
    rospy.init_node("main_program", anonymous = True)
    pub = rospy.Publisher('speech', Int8, queue_size = 20)
    pub_vel = rospy.Publisher('/temp_cmd_vel', Twist, queue_size = 10)
    pub_speech = rospy.Publisher('isKeepGoing', Float64MultiArray, queue_size = 10)
    rospy.Subscriber ('/odom', Odometry, odom_callback)#要去查一下要訂閱哪個node可以得到現在的角度
    rospy.loginfo("hello")
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        for i in range(10):
            rospy.Subscriber('target_floor', Float64MultiArray, targetFloor)#從"speech的模組"發訊號到node"target_floor"，要傳 目標的樓層 與 目標的房間，用array傳送
            rospy.sleep(0.1)
        if(target_floor != 1):
            rospy.loginfo("searching for elevator")
            while(1):#只找電梯
                rospy.Subscriber('image',Float64MultiArray,ImageDepth)#將"影像處理模組"處理完的影像傳到node"image"，傳送深度的array
                rospy.Subscriber('SeeSomething', Int8, SeeSomething)#如果有看到東西，往node"SeeSomething"publish，1代表有看到電梯，2代表看到殘障標誌
                #3代表看到門，4代表看到門牌

                if(ISee == 1):
                    rospy.loginfo("see the elevator")
                    move_straight_time("forward", 1, 1)#往電梯方向移動一段距離 # 還要更改
                    while(1):
                        rospy.Subscriber('SeeSomething', Int8, SeeSomething)
                        if(ISee == 2):
                            rospy.loginfo("see the sign")
                            move_straight_time("forward", 1, 1)#移動到電梯前 # 還要更改
                            mention = 1
                            pub.publish(mention)#跟speech模組說應該要提醒使用者按電梯面板呼叫電梯了
                            rospy.loginfo("break1")
                            break
                    while(1):
                        rospy.Subscriber('HeardSomething', Int8, HeardSomething)#1代表聽到電梯到了，2代表聽到電梯門要關了，3代表到達目標樓層，可以往前走了
                        rospy.loginfo("Waiting for the elevator")

                        # rospy.Subscriber()#訂閱lidar發出的那個node
                        if(IHeard == 1):# & 電梯前面深度夠深):
                            rospy.loginfo("elevator arrived")
                            move_straight_time("forward", 1, 1)#往前移動
                            mention = 2
                            pub.publish(mention)#跟speech模組說應該要提醒使用者左後方電梯了
                            while(1):
                                rospy.Subscriber('HeardSomething', Int8, HeardSomething)
                                rospy.loginfo("Waiting for the door closed")
                                
                                if(IHeard == 2):
                                    rospy.loginfo("elevator door closed")
                                    # rotate(-90)#向右轉向
                                    # rospy.sleep(1)
                                    # move_straight_time("forward", 1, 1)#向前走一段
                                    # rospy.sleep(1)
                                    # rotate(90)#向左轉向，上面三段走到門口
                                    # rospy.sleep(1)
                                    waitForEleOpen = 1
                                    rospy.loginfo("break2")
                                    break
                        if(waitForEleOpen == 1):
                            rospy.loginfo("break3")
                            break
                        rospy.sleep(1)
                    while(1):
                        rospy.Subscriber('HeardSomething', Int8, HeardSomething)
                        rospy.loginfo("Waiting for reaching the floor")
                        if(IHeard == 3):#還要跟影像判斷
                            rospy.loginfo("reach the target floor")
                            movemode = 1
                            mention = 3
                            pub.publish(mention)
                            move_straight_time("forward", 1, 1)#往前走
                            rospy.loginfo("break4")
                            break
                else:
                    rospy.loginfo("i am here")
                    threshold = 100 #lidar中可以走的最短距離 #還要更改

                    if(threshold > 10000):#lidar中的資訊):#還要更改
                        rospy.loginfo("no way in front")

                        # 轉180度 #繼續用影像找路

                    else:
                        rospy.loginfo("keep finding the elevator")
                        rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, callback=depth_callback, queue_size=1)
    
                if(movemode == 1):
                    rospy.loginfo("break5")
                    break
                rospy.sleep(1)

        # 不用找電梯 or 找到電梯了
        while(1):
            rospy.loginfo("Now is finding door mode")
            rospy.Subscriber('SeeSomething', Int8, SeeSomething)#如果有看到東西，往node"SeeSomething"publish，1代表看到電梯，2代表看到殘障標誌
                #3代表看到門，4代表看到門牌
            if(ISee == 3):
                rospy.loginfo("see the door")
                move_straight_time("forward", 1, 1)#往前移動到門前 #還要更改
                rospy.Subscriber('SeeSomething', Int8, SeeSomething)
                rospy.sleep(4)
                if(ISee == 4):
                    rospy.loginfo("see the door sign")
                    rospy.Subscriber('SeeDoorNum', Int8, SeeDoorNum)#從node"SeeDoorNum"拿現在觀察到的門牌號碼
                    rospy.sleep(1)
                    rospy.loginfo(door_num.data[1])
                    if(target_room == door_num.data[1]):
                        rospy.loginfo("we find the target room")
                        mention = 4
                        pub.publish(mention)
                        targetAchieve = 1
                        rospy.loginfo("break7")
                        break
                    else:
                        rospy.loginfo("not target room")
                        pub_speech.publish(door_num)#發訊息到node"isKeepGoing"，由"speech模組"訂閱，詢問要不要繼續前進
                        door_num.data[0] = door_num.data[1]#0代表之前的門牌號碼，1代表現在看到的門牌號碼
                        
                        rospy.Subscriber("isKeepGoing_reply", Int8, isKeepGoing)
                        if(keepGoing == 0):#現在走的方向是不對的
                            rospy.loginfo("not on the right way")
                            
                            #轉回本來行徑方向
                            #轉180度
                            move_straight_time("forward", 1, 1)#走一小段距離
                            
                        if(keepGoing == 1):#現在走的方向是對的
                            rospy.loginfo("on the right way")
                            #轉回本來行徑的方向 
                            move_straight_time("forward", 1, 1)#走一小段距離

                else:
                    rospy.loginfo("don't see door sign")
                    #轉回本來行徑方向
                    move_straight_time("forward", 1, 1)#往前走一小段距離
                    # 沒看到門牌，繼續用影像找路
            else:
                threshold = 100#lidar中可以走的最短距離 # 還要更改

                if(threshold > 10000):#lidar中的資訊):#還要更改
                    rospy.loginfo("no way in front")
                    # 轉180度 #繼續用影像找路

                else:
                    rospy.loginfo("keep finding door")
                    rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, callback=depth_callback, queue_size=1)
            rospy.sleep(2)            
        if targetAchieve==1:
            rospy.loginfo("oh break")
            break