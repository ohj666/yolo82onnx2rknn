# 导出 onnx
import torch

from ultralytics import YOLO
# model = torch.load('yolov8n.pt')
model = YOLO('yolov8_relu.pt')
success = model.export(format="onnx")
