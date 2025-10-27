import lightly_train
from torchvision import utils, io
import matplotlib.pyplot as plt

model = lightly_train.load_model_from_checkpoint(
    checkpoint="weights/ltdetr_convnext-tiny_coco.ckpt",
)

test_image = "detr_test.jpg"

labels, boxes, scores = model.predict(test_image).values()

image_with_boxes = utils.draw_bounding_boxes(
    image=io.read_image(test_image),
    boxes=boxes,
    labels=[model.classes[i.item()] for i in labels],
)

fig, ax = plt.subplots(figsize=(30, 30))
ax.imshow(image_with_boxes.permute(1, 2, 0))
fig.savefig(f"predictions.png")