import sys
# sys.path.insert(0, './YOLOX')
sys.path.insert(1,'./3.mp4')
import torch
import cv2

import time

import numpy as np
from collections import deque
from collections import Counter

# importing Detector
from ultralytics import YOLO

# Importing Tracker
from deep_sort.utils.parser import get_config
from bytetrack.byte_tracker import BYTETracker

# Importing Visuals
from visuals import *

from intersect_ import *

import math

import datetime

# A Dictionary to keep data of tracking
data_deque = {}
speed_dict = {}
point_start = (0,0)
point_end = (0,0)

time_test = {}
flag = True
# class_names = COCO_CLASSES
lines  = [
    {'Title' : 'North', 'Cords' : [(1720, 550), (1119, 480)]},
    {'Title' : 'South', 'Cords' : [(625, 727), (1532, 861)]},
    {'Title' : 'East', 'Cords' : [(1764, 595), (1731, 806)]},
    {'Title' : 'West', 'Cords' : [(915, 505), (586, 657)]}
]

object_counter = {
    'North' : Counter(),
    'South' : Counter(),
    'East'  : Counter(),
    'West'  : Counter(),
}


pts = {}

def vis_track(img, outputs):
    outputs = np.array(outputs)
    if len(outputs) == 0:
        return img

    for key in list(pts):
        if key not in outputs[:,-2]:
            pts.pop(key)

    for i in range(len(outputs)):
        box = outputs[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        id = box[4]
        clsid = box[5]

        if id not in pts:
            pts[id] = deque(maxlen=64)

        # pts = { '1' : deque(),'2' : deque()}

        center = (int((x0+x1)/2) , int((y0+y1)/2))
        pts[id].append(center)

        # Drawing a circle
        color = compute_color_for_labels(clsid)
        thickness = 5
        cv2.circle(img,  (center), 1, color, thickness)

        # Draw motion path
        for j in range(1, len(pts[id])):
            if pts[id][j - 1] is None or pts[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 3)
            cv2.line(img,(pts[id][j-1]), (pts[id][j]),(color),thickness)

    return img



def estimateSpeed(location1, location2, time = 0):

    height = location1[0] - location2[0]
    width = location1[1] - location2[1]
    
    distance_in_pixels = math.sqrt(math.pow(height,2) + math.pow(width,2))

    pixels_per_meter = 0.0002645833 

    distance_in_meters = distance_in_pixels*pixels_per_meter
    distance_in_km = distance_in_meters
    fps = 30 
    Time_  = 1/fps
    if time >0: 
         
        speed_mps = distance_in_km/time
    else:  speed_mps = distance_in_meters/Time_
    
    speed_kmph = speed_mps*3.6

    return int(speed_kmph)





#Draw the Lines
def draw_lines(lines, img):
    for line in lines:
        img = cv2.line(img, line['Cords'][0], line['Cords'][1], (255,255,255), 3)
    return img

# Update the Counter
def update_counter(centerpoints, obj_name, id,point_start, point_end,flag):
    time_taken = 0 
    data = []
    for line in lines:
        p1 = Point(*centerpoints[0])
        q1 = Point(*centerpoints[1])
        p2 = Point(*line['Cords'][0])
        q2 = Point(*line['Cords'][1])
        if point_online(centerpoints[0],line['Cords'][0],line['Cords'][1]): 
            time_start = np.round(time.time(),3)
        
            if flag:
                time_test.update({id:time_start})   
                point_start = centerpoints[0]
                flag = False
        if intersect(centerpoints[0], centerpoints[1], line['Cords'][0], line['Cords'][1]): 
            point_end = centerpoints[1]
            if id in time_test:    
                time_taken = np.round(time.time(),3)-time_test.get(id)
                
            object_counter[line['Title']].update([obj_name])
            speed = estimateSpeed(location1 = point_start, location2 = point_end,time = time_taken)
       
            speed_dict[id] = speed
            flag = True
            print("intersection detected")
            data.append({
                'Category' : obj_name,
                'direction': line['Title'],
                'Time'     : datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Speed'    : speed,
                'id'       : id
                    }) 
            
    return data

# Draw the Final Results
def draw_results(img):
    x = 100
    y = 100
    offset = 50
    for line_name, line_counter in object_counter.items():
        Text = line_name + " : " + ' '.join([f"{label}={count}" for label, count in line_counter.items()])
        cv2.putText(img, Text, (x,y), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        y = y+offset
    return img



# Function to calculate delta time for FPS when using cuda
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

# Draw the boxes having tracking indentities 
def draw_boxes(names,img, bbox, object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape 
    # Cleaning any previous Enteries
    [data_deque.pop(key) for key in set(data_deque) if key not in identities]
    [speed_dict.pop(key) for key in set(data_deque) if key not in identities]
    
    frame_data = []
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) +offset[0]  for i in box]  
        box_height = (y2-y1)
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
        id = int(identities[i]) if identities is not None else 0

        if id not in set(data_deque):  
          data_deque[id] = deque(maxlen= 100)

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '%s' % (obj_name)
        
        data_deque[id].appendleft(center) #appending left to speed up the check we will check the latest map

        if len(data_deque[id]) >=2:
            data = update_counter(centerpoints = data_deque[id],point_start= point_start, point_end= point_end, flag = flag,obj_name = obj_name, id = id)
            frame_data.extend(data)

        if id in speed_dict:
            speed = speed_dict[id]
        else:
            speed = ''
        
        UI_box(box, img, label=label + str(speed) + 'km/h', color=color, line_thickness=3, boundingbox=True)
        

    return img, frame_data

# Tracking class to integrate Deepsort tracking with our detector
class Tracker():
    def __init__(self, classes, filter_classes=None, ckpt='wieghts/yolox_s.pth'):

        self.detector = YOLO(ckpt)
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        self.names = self.detector.names
        self.classes = classes

        cfg = get_config()
        cfg.merge_from_file("D:/Tunf/NOO/DATN/Traffic-Flow-Analysis/bytetrack/configs/bytetrack.yaml")
        self.bytetrack = BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=cfg.bytetrack.frame_rate)
        self.filter_classes = filter_classes
    def update(self, image):
        height, width, _ = image.shape 
        info = self.detector.predict(source=image, classes =self.classes, device = self.device)[0]
        outputs = []

        if len(info.boxes)>0:
            # scores = []
            # objectids = []
            # bbox_xywh = []
            dets=[]
            for pred in info:
                x1,y1,x2,y2 = pred.boxes.xyxy.int().tolist()[0]
                conf = pred.boxes.conf.item()
                cls = pred.boxes.cls.item()
              
                detection = [int(x1)] +[int(y1)] +[int(x2)] +[int(y2)] + [conf] + [cls]
                dets.append(detection) 
                
            # print(dets)
            # bbox_xywh = torch.Tensor(bbox_xywh)
            dets = torch.Tensor(dets)
           
            outputs = self.bytetrack.update(dets.cpu(),image)
            # print("*"*20)
            # print(outputs)

            if len(outputs) > 0:


                bbox_xyxy =np.array(outputs)[:, :4]
                identities =np.array(outputs)[:, 4]
                object_id =np.array(outputs)[:, 5]
                image,frame_data = draw_boxes(self.names,image, bbox_xyxy, object_id,identities)

            else:
                frame_data = []
        else: frame_data = []
            
            
        return image, outputs, frame_data

