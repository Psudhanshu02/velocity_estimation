from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler
import glob
import pathlib
from utils import *
import imutils
import time




device = torch.device(f"cuda:{0}")

from model import DepthVelocityEstimationNet
model=DepthVelocityEstimationNet(3, depth=20)
parameters=torch.load("runs/_25.pt")
model.load_state_dict(parameters)



from torch.utils.data import (
	Dataset, 
	DataLoader
)


preprocess_pipeline = ProcessPipeline([
                                       agcwd,
                                       apply_kernel, 
                                       crop, 
                                       resize], 
                            [
                      		{"w": 0.4},
                            {"kernel": PreProcessingDefaults.sharpening_kernel_2},
                            {'crop_points': (
                                PreProcessingDefaults.crop_x, 
                                PreProcessingDefaults.crop_y),
                             "color": True},
                            {"factor": 1.2}])

def preprocess_images(images):
    i = 0
    for f in images:
        s = pathlib.Path(f).stem
        try:
            x = cv2.imread(f"data/predict/{s}.jpg")
            x = preprocess_pipeline.process_image(x)
            cv2.imwrite(f"data/predict/{s}.jpg", x)
            i += 1
            if i % 100 == 0:
                print(f"{i}", flush=True)   
        except:
            print(f"woops on {f}", flush=True)
            raise

images = glob.glob(f"data/predict/*.jpg")
images = sorted(images, key=lambda x: int(pathlib.Path(x).stem))
# Create a VideoCapture object
cap = cv2.VideoCapture('data/test.mp4')


    


# Check if camera opened successfully
if not cap.isOpened(): 
    print("Error opening video file")

# Default resolutions of the frame are obtained. Convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


arr=[]
frame_count=1
count=1

mean_result=0

x1s=[]
x2s=[]

while(cap.isOpened() ):
  
  ret, frame = cap.read()
  
  if ret == True:
        frame=cv2.resize(frame,(640,480))

        if count==22:
            x1 = torch.stack([x for x in x1s], 0)
            x2=torch.stack([x for x in x2s], 0)

            x1=x1.unsqueeze(0)
            x2=x2.unsqueeze(0)
    
            model=model.to(device)
            result=model(x1,x2)
            mean_result=torch.mean(result).item()
            
            x1s=[]
            x2s=[]
            count=1
        

        
        frame=preprocess_pipeline.process_image(frame)

        _x1=torch.tensor(frame).float().to(device)
        _x1=_x1.T
        if count!=21:
            x1s.append(_x1)
        
        if count !=1:
            x2s.append(_x1)

        count+=1
        text="speed :"+str(round(mean_result*18/5))+"km/h"
        arr.append(text)
        
        
        
        cv2.putText(frame, text,(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,cv2.LINE_AA)
        cv2.imshow('video',frame)
        print(frame_count)
        frame_count+=1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
  else:
     break
 
