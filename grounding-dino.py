import requests
import torch
from PIL import Image
from PIL import ImageDraw

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
model_id = "./weights" # IDEA-Research/grounding-dino-base
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
image = Image.open('./test06.png').convert('RGB')
text = "a car"
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    #box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")

draw = ImageDraw.Draw(image)
for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    draw.rectangle(box.tolist(), outline="red", width=3)
    label_text = f"{label}: {score:.3f}"
    draw.text((box[0], box[1]), label_text, fill="red")

image.show()
image.save("detection_result.jpg")