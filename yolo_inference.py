from ultralytics import YOLO

model = YOLO("models/best2.pt")  # load a pretrained model (recommended for training)
results = model.predict(source="input_videos/08fd33_4.mp4",save=True)  

print(results[0])  # print results to terminal
print("================================")
for box in results[0].boxes:
    print(box)  # print box details to terminal
