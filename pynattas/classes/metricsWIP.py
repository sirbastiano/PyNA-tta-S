## This file is a work in progress and is not yet functional

import torch
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class MAPCalculator:
    def __init__(self, model, test_dataloader, iou_threshold=0.5, conf_threshold=0.5):
        self.model = model
        self.test_dataloader = test_dataloader
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def compute_map(self):
        self.model.eval()
        coco_gt = COCO(self.test_dataloader.dataset.ann_file)
        coco_dt = coco_gt.loadRes(self.generate_predictions())
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.iouThrs = [self.iou_threshold]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP = coco_eval.stats[0]  # AP@IoU=0.50
        return mAP

    def generate_predictions(self):
        results = []

        for images, targets in self.test_dataloader:
            images = images.cuda()
            with torch.no_grad():
                outputs = self.model(images)

            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                preds = self.post_process(output)

                for pred in preds:
                    bbox = pred[:4].tolist()
                    score = pred[4].item()
                    category_id = pred[5:].argmax().item() + 1  # Assuming category indices start from 1

                    result = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "score": score,
                    }
                    results.append(result)

        return results

    def post_process(self, preds):
        objectness_scores = preds[..., 4]
        keep = objectness_scores > self.conf_threshold
        preds = preds[keep]

        bboxes = preds[..., :4]
        scores = preds[..., 4]
        classes = preds[..., 5:].argmax(dim=-1)

        keep = nms(bboxes, scores, self.iou_threshold)
        preds = preds[keep]

        return preds

# Helper functions
def iou_single(box1, box2):
    inter_rect_x1 = max(box1[0], box2[0])
    inter_rect_y1 = max(box1[1], box2[1])
    inter_rect_x2 = min(box1[2], box2[2])
    inter_rect_y2 = min(box1[3], box2[3])

    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)
    b1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    b2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-9)
    return iou

def scale_boxes(boxes, img_size):
    width, height = img_size
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    return boxes


'''
import torch
from metrics.mAP import MAPCalculator
from pytorch_lightning import Trainer
from my_test_datamodule import MyTestDataModule  # Replace with your actual DataModule

# Load your trained model
model = YourModel.load_from_checkpoint("path/to/your/checkpoint.ckpt")
model.cuda()

# Initialize the test datamodule
test_dm = MyTestDataModule()
test_dm.setup('test')

# Create the mAP calculator instance
map_calculator = MAPCalculator(model, test_dm.test_dataloader(), iou_threshold=0.5, conf_threshold=0.5)

# Compute mAP
mAP = map_calculator.compute_map()
print(f"mAP@50: {mAP}")
'''