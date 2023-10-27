from ultralytics import YOLO
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = YOLO('yolov8s.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
model.train(data='coco128.yaml', epochs=3, workers=0, line_thickness=3)
print(model)
model.export(format='onnx')