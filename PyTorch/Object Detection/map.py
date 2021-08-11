import torch
from iou import intersection_over_union
from collections import Counter


def mean_average_precision(boxes_preds, boxes_labels, iou_threshold=0.5, box_format='corners', num_classes=20):

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in boxes_preds:
            if detection[1] == c:
                detections.append(detection)

        for label in boxes_labels:
            if label[1] == c:
                ground_truths.append(label)


        # returns dictionary of train_idx (gt[0]) and how many times it occurs, in other words number of bboxes for each image
        amount_bboxes = Counter(gt[0] for gt in ground_truths)

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections = sorted(detections, key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate((detections)):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                            torch.tensor(gt[3:]),
                                            box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0][best_gt_idx]] == 0:
                    TP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = toch.cumsum(FP, dim=0)

        recalls = TP_cumsum/ (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon)) 
        precisions = torch.cat(torch.tensor([1]), precisions)
        recalls = torch.cat(torch.tensor([0]), recalls)

        average_precisions.append(torch.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)




