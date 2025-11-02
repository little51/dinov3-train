from ultralytics import YOLO

model_before = YOLO("out/my_experiment/exported_models/exported_last.pt")
results_before = model_before.val(data="coco8.yaml")
print(f"微调前mAP: {results_before.box.map}")
model_after = YOLO("runs/detect/train/weights/best.pt")
results_after = model_after.val(data="coco8.yaml")
print(f"微调后mAP: {results_after.box.map}")