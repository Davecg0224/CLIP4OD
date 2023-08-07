#!/usr/bin/env python
import rospy
import rospkg
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import String, Float64MultiArray, Int8
from cv_bridge import CvBridge

import torch
import clip
from PIL import Image
import json
import numpy as np
import cv2
import torchvision 
from torchvision import transforms
from torchvision.ops import nms

def RPN(img, model, detectThres = 0.5, iouThres = 0.3):
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(img)

    model.eval()
    preds = rpnModel([img_tensor])[0]
    boxes = preds['boxes']
    scores = preds['scores']
    boxes = boxes[scores >= detectThres]
    scores = scores[scores >= detectThres]
    final_boxesId = nms(boxes = boxes, scores = scores, iou_threshold = iouThres)
    final_boxes = boxes[final_boxesId]
    return final_boxes, scores

def CLIPprocess(img_crop, text):
    image = preprocess(img_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        id = np.argmax(probs)
    return probs, id

def buildOptions(target):
    targetFloor = target[0]
    targetList = [targetFloor + str(num).rjust(2, "0") for num in range(1, 40)]
    return targetList

def getCenterAndPub(box):
    HFov = 110
    halfW = Rw / 2
    x1, y1, x2, y2 = box
    x = np.mean((x1, x2))
    y = np.mean((y1, y2))
    angle = -(round(x) - halfW) / halfW * HFov/2 
    array = Float64MultiArray(data = [x, y, angle])
    pubGoal.publish(array)
    print("Degree to turn:", angle)
    print('Object center point: ', [x, y])

def webcam_callback(web_img):
    cv_web_image = bridge.compressed_imgmsg_to_cv2(web_img)
    web_img_pil = Image.fromarray(cv_web_image)
    boxes, rpnScores = RPN(web_img_pil, rpnModel)
    if len(boxes) == None:
        print("No proper region to pay attention")
    else: 
        print("Number of boxes: ", len(boxes))
        for box in boxes.detach().numpy():
            x1, y1, x2, y2 =  box
            web_img_crop = web_img_pil.crop((int(x1), int(y1), int(x2), int(y2)))
            web_img_crop = preprocess(web_img_crop).unsqueeze(0).to(device)
            sceneProbs, sceneId = CLIPprocess(web_img_crop, tokenScene)
            conf = np.max(sceneProbs)
            if  conf > classThres:
                if sceneList[sceneId] == 'doorplate':
                    targetProbs, targetId = CLIPprocess(web_img_crop, tokenTarget)
                    targetConf = np.max(targetProbs)
                    print(targetProbs[0][11])
                    if targetConf > classThres:
                        if targetList[targetId] == '412':
                        # if targetProbs[0][11] > 0.025:
                            print("Reach the target room")
                        else:
                            print("Keep searching")

                        print("=====================================")

def detecting(cameraImage):
    if cameraImage != None:
        cv_image = bridge.compressed_imgmsg_to_cv2(cameraImage)
        # rospy.loginfo("Now searching the scene")
        img_pil = Image.fromarray(cv_image)

        # Region Proposal
        boxes, rpnScores = RPN(img_pil, rpnModel)

        if len(boxes) == None:
            print("No proper region to pay attention")
        else: 
            print("Number of boxes: ", len(boxes))
            for box in boxes.detach().numpy():
                x1, y1, x2, y2 =  box
                img_crop = img_pil.crop((int(x1), int(y1), int(x2), int(y2)))
                # image = preprocess(img_crop).unsqueeze(0).to(device)
                objProbs, objId = CLIPprocess(img_crop, tokenScene)
                conf = np.max(objProbs)
                if  conf > classThres:
                    text = '{}({:.1f}%)'.format(objList[objId], conf*100)
                    rospy.loginfo(text)
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (20, 20, 225), 2)
                    cv2.putText(cv_image, text, 
                                (int(x1), int(y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 52, 52), 
                                2, lineType=cv2.LINE_AA)

                    print("=====================================")

        # cv_image = cv2.resize(cv_image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imshow("vision", cv_image) 
        cv2.waitKey(1)

def image_callback(msg):
    global cameraImage
    cameraImage = msg
    # rospy.loginfo("Recieve image")

cameraImage = None

if __name__ == '__main__':
    rospy.loginfo("Hello Bro!!")
    print("Hello Bro!!")
    
    # ## config 
    Rw = 640
    Rh = 360
    # CLIP model: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    selectModel = "ViT-B/16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(selectModel, device=device) 
    classThres = 0.5
    filePath = rospkg.RosPack().get_path('detection')

    with open(filePath + '/src/detect.json') as f:
        cfg = json.load(f)

    objDict = cfg['objList']
    objList = list(objDict.values())
    # tokenScene= clip.tokenize(objList).to(device)
    tokenScene = torch.cat([clip.tokenize(f"a photo of the {c}") for c in objList]).to(device)
    
    # target = '412'
    # targetList = buildOptions(target)
    # tokenTarget = torch.cat([clip.tokenize(f"numbers with {c}") for c in targetList]).to(device)

    ## avaiable rpn model: [fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn]
    rpnModel = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights = 'DEFAULT')    

    ##################################################################
    ## zed2 input
    bridge = CvBridge()
    # pubDetect = rospy.Publisher("SeeSomething", Int8, queue_size = 1)
    # pubDoorNum = rospy.Publisher("SeeDoorNum", Int8, queue_size = 1)
    # pubGoal = rospy.Publisher("CenterAndDegree", Float64MultiArray, queue_size = 1)

    rospy.init_node('detect_node', anonymous=True)
    while not rospy.is_shutdown():
        rospy.Subscriber("/camera/compressed", CompressedImage, image_callback, queue_size = 1, buff_size = 65565*1024)
        ## webcam for checking room number
        # rospy.Subscriber("webcam_img", CompressedImage, image_callback, queue_size = 1, buff_size = 65565*1024)
        detecting(cameraImage)
        # rospy.spin()
