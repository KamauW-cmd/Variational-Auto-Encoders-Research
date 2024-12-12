import ultralytics
from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter 
from PIL import Image
import cv2 
import matplotlib as plt

'''
model = YOLO("/home/kamau/runs/detect/train49/weights/best.pt")
results = model.train(data = '/home/kamau/Project/Images/Cups/config.yaml', epochs = 2000)
'''

model = YOLO('/home/kamau/runs/detect/train52/weights/best.pt')
result = model.predict('/home/kamau/Test_Images/mug5.jpg', save = True, save_txt = True)
