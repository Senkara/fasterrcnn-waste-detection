import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import NUM_CLASSES


def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model