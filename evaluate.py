import os
import json
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from config import BASE_DIR, FOLDS, CLASS_NAMES, DEVICE
from dataset import YOLODetectionDataset
from model import get_model
from utils import collate_fn


RESULTS_DIR = "eval_results"
IOU_THRESHOLD = 0.50
SCORE_THRESHOLD = 0.50

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_val_paths(fold):
    train_dir = os.path.join(BASE_DIR, fold, "train")
    val_dir = os.path.join(BASE_DIR, fold, "val")
    valid_dir = os.path.join(BASE_DIR, fold, "valid")

    if os.path.exists(val_dir):
        eval_dir = val_dir
    elif os.path.exists(valid_dir):
        eval_dir = valid_dir
    else:
        raise FileNotFoundError(f"{fold} içinde ne 'val' ne de 'valid' klasörü bulundu.")

    images_dir = os.path.join(eval_dir, "images")
    labels_dir = os.path.join(eval_dir, "labels")

    return images_dir, labels_dir


def box_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                 iou_threshold=0.5, score_threshold=0.5):
    """
    Object detection için object-level eşleştirme.
    Background class index = len(CLASS_NAMES)
    """
    bg_idx = len(CLASS_NAMES)

    # düşük skorlu prediction'ları at
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    # skor sırasına göre prediction'ları sırala
    if len(pred_scores) > 0:
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

    gt_matched = set()
    pred_matched = set()

    y_true = []
    y_pred = []

    # prediction -> en iyi gt eşleşmesi
    for p_idx in range(len(pred_boxes)):
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx in range(len(gt_boxes)):
            if g_idx in gt_matched:
                continue

            iou = box_iou(pred_boxes[p_idx].tolist(), gt_boxes[g_idx].tolist())
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx != -1 and best_iou >= iou_threshold:
            gt_matched.add(best_gt_idx)
            pred_matched.add(p_idx)

            y_true.append(int(gt_labels[best_gt_idx].item()))
            y_pred.append(int(pred_labels[p_idx].item()))
        else:
            # false positive
            y_true.append(bg_idx)
            y_pred.append(int(pred_labels[p_idx].item()))

    # eşleşmeyen gt -> false negative
    for g_idx in range(len(gt_boxes)):
        if g_idx not in gt_matched:
            y_true.append(int(gt_labels[g_idx].item()))
            y_pred.append(bg_idx)

    return y_true, y_pred


def save_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def evaluate_fold(fold):
    print(f"\nEvaluating {fold}...")

    model_path = f"{fold}_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} bulunamadı.")

    val_images, val_labels = get_val_paths(fold)
    dataset = YOLODetectionDataset(val_images, val_labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        backend="faster_coco_eval"
    )

    all_true = []
    all_pred = []

    for images, targets in loader:
        # Bazı durumlarda images tuple-of-tuples gibi gelebiliyor
        images = list(images)
        targets = list(targets)

        fixed_images = []
        for img in images:
            if isinstance(img, (tuple, list)):
                img = img[0]
            fixed_images.append(img.to(DEVICE))

        fixed_targets = []
        for t in targets:
            if isinstance(t, (tuple, list)):
                t = t[0]
            fixed_targets.append(t)

        outputs = model(fixed_images)

        preds_for_map = []
        targets_for_map = []

        for output, target in zip(outputs, fixed_targets):
            # model labels: 1..N  -> mAP için 0..N-1
            pred_boxes = output["boxes"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()
            pred_labels = output["labels"].detach().cpu() - 1

            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu() - 1

            preds_for_map.append({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            })

            targets_for_map.append({
                "boxes": gt_boxes,
                "labels": gt_labels
            })

            y_true, y_pred = greedy_match(
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                iou_threshold=IOU_THRESHOLD,
                score_threshold=SCORE_THRESHOLD
            )

            all_true.extend(y_true)
            all_pred.extend(y_pred)

        metric.update(preds_for_map, targets_for_map)

    map_results = metric.compute()

    label_names_with_bg = CLASS_NAMES + ["background"]
    label_ids_with_bg = list(range(len(label_names_with_bg)))

    cm = confusion_matrix(all_true, all_pred, labels=label_ids_with_bg)

    report = classification_report(
        all_true,
        all_pred,
        labels=label_ids_with_bg,
        target_names=label_names_with_bg,
        zero_division=0,
        output_dict=True
    )

    fold_result_dir = os.path.join(RESULTS_DIR, fold)
    os.makedirs(fold_result_dir, exist_ok=True)

    # confusion matrix görseli
    save_confusion_matrix(
        cm,
        label_names_with_bg,
        os.path.join(fold_result_dir, "confusion_matrix.png")
    )

    # confusion matrix raw
    np.savetxt(
        os.path.join(fold_result_dir, "confusion_matrix.csv"),
        cm,
        fmt="%d",
        delimiter=","
    )

    # mAP sonuçları
    map_json = {
        "map": float(map_results["map"].item()),
        "map_50": float(map_results["map_50"].item()),
        "map_75": float(map_results["map_75"].item()),
        "mar_100": float(map_results["mar_100"].item()),
    }

    if "classes" in map_results and "map_per_class" in map_results:
        classes = map_results["classes"].tolist()
        map_per_class = map_results["map_per_class"].tolist()
        map_json["map_per_class"] = {
            CLASS_NAMES[int(cls_id)]: float(ap)
            for cls_id, ap in zip(classes, map_per_class)
            if int(cls_id) < len(CLASS_NAMES)
        }

    with open(os.path.join(fold_result_dir, "map_results.json"), "w", encoding="utf-8") as f:
        json.dump(map_json, f, indent=4)

    with open(os.path.join(fold_result_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"{fold} bitti.")
    print("mAP      :", map_json["map"])
    print("mAP@0.50 :", map_json["map_50"])
    print("mAP@0.75 :", map_json["map_75"])

    return {
        "fold": fold,
        "map": map_json["map"],
        "map_50": map_json["map_50"],
        "map_75": map_json["map_75"],
        "report": report
    }


def summarize_results(results):
    summary = {
        "map_mean": float(np.mean([r["map"] for r in results])),
        "map_50_mean": float(np.mean([r["map_50"] for r in results])),
        "map_75_mean": float(np.mean([r["map_75"] for r in results]))
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n=== ORTALAMA SONUÇLAR ===")
    print("Mean mAP      :", summary["map_mean"])
    print("Mean mAP@0.50 :", summary["map_50_mean"])
    print("Mean mAP@0.75 :", summary["map_75_mean"])


def main():
    all_results = []
    for fold in FOLDS:
        result = evaluate_fold(fold)
        all_results.append(result)

    summarize_results(all_results)


if __name__ == "__main__":
    main()