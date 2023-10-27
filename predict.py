from ultralytics import YOLO
model = YOLO(r'D:\git_project\ultralytics\runs\detect\train2\weights\best.pt')
print(model)