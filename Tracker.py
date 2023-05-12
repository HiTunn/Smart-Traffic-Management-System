import sys
# sys.path.insert(0, './YOLOX')
import torch
import cv2
# from yolox.utils import vis
import time
# from yolox.exp import get_exp
import numpy as np
from collections import deque

# importing Detector
# from yolox.data.datasets.coco_classes import COCO_CLASSES
# from detector import Predictor

# Importing Deepsort

from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort
from bytetrack.byte_tracker import BYTETracker
# from ultralytics.nn.autobackend import AutoBackend
from ultralytics import YOLO

# Importing Visuals
from visuals import *

# A Dictionary to keep data of tracking
data_deque = {}

# class_names = COCO_CLASSES

# Function to calculate delta time for FPS when using cuda
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# Draw the boxes having tracking indentities 
def draw_boxes(names,img, bbox, object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape 
    # Cleaning any previous Enteries
    [data_deque.pop(key) for key in set(data_deque) if key not in identities]

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
        UI_box(box, img, label=label + str(id), color=color, line_thickness=3, boundingbox=True)

    return img

# Tracking class to integrate Deepsort tracking with our detector

class Tracker():
    def __init__(self,classes, frame_rate,filter_classes=None, ckpt='D:/Tunf/NOO/DATN/yolov8_tracking/weights/best_v8n.pt'):


        self.detector = YOLO(ckpt)
        self.names = self.detector.names
        self.classes = classes
        cfg = get_config()
        cfg.merge_from_file("D:/Tunf/NOO/DATN/Traffic-Flow-Analysis/bytetrack/configs/bytetrack.yaml")
        self.bytetrack = BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=frame_rate)
        self.filter_classes = filter_classes
    def update(self, image, visual = True, logger_=True):
        height, width, _ = image.shape 
        info = self.detector.predict(source=image, classes =self.classes)[0]
        outputs = []
        
        if len(info.boxes)>0:
            bbox_xywh = []
            scores = []
            objectids = []
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
            data = []
            if len(outputs) > 0:
                if visual:
                    if len(outputs) > 0:
                        bbox_xyxy =np.array(outputs)[:, :4]
                        identities =np.array(outputs)[:, 4]
                        object_id =np.array(outputs)[:, 5]
                        image = draw_boxes(self.names,image, bbox_xyxy, object_id,identities)
            return image, outputs
        else: 
            return image, None


if __name__=='__main__':
    
        
    # D:/Tunf/NOO/DATN/data_test/bikes.mp4 
    # D:/Tunf/NOO/DATN/Traffic-Flow-Analysis/3.mp4
    cap = cv2.VideoCapture("D:/Tunf/NOO/DATN/Traffic-Flow-Analysis/3.mp4") 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # D:/Tunf/NOO/DATN/yolov8_tracking/weights/best_v8n.pt
    tracker = Tracker(classes = [2, 3, 5, 7],filter_classes=None,frame_rate = fps, ckpt='weights/yolov8n.pt')    # instantiate Tracker
    
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))

    vid_writer = cv2.VideoWriter(
        f'D:/Tunf/NOO/DATN/Traffic-Flow-Analysis/track_demo_.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    ) # open one video
    frame_count = 0
    fps = 0.0
    while True:
        ret_val, frame = cap.read() # read frame from video
        
        t1 = time_synchronized()
        if ret_val:
            frame, bbox = tracker.update(frame, visual=True, logger_=False)  # feed one frame and get result
            vid_writer.write(frame)
            cv2.imshow("a",frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                
                break
            fps  = ( fps + (1./(time_synchronized()-t1)) ) / 2
        else:
            break

    cap.release()
    # vid_writer.release()
    cv2.destroyAllWindows()
