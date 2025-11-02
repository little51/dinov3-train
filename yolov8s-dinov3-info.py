from ultralytics import YOLO

model = YOLO("out/my_experiment/exported_models/exported_last.pt")
    
print("任务头关键信息:")
print(f"• 检测类别数: {model.model.nc}")
print(f"• 模型任务类型: {getattr(model.model, 'task', '未知')}")