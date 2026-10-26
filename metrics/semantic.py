import os
import torch
import logging
import numpy as np
from torchmetrics.classification import MulticlassConfusionMatrix
from torch_scatter import scatter_add
from torch import Tensor

class BaseMetricResults(dict):


    def __getattr__(self, key: str) -> Tensor:


        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: Tensor) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self:
            del self[key]
        raise AttributeError(f"No such attribute: {key}")

class SemanticMetricResults(BaseMetricResults):

    __slots__ = (
        'oa',
        'macc',
        'miou',
        'iou_per_class',
        'seen_class')


class ConfusionMatrix(MulticlassConfusionMatrix):


    def __init__(self, num_classes, **kwargs):
        super().__init__(
            num_classes,
            ignore_index=num_classes,
            normalize=None,
            validate_args=False,
            **kwargs)

    def update(self, pred, target):

        assert not target.is_floating_point()
        assert pred.shape[0] == target.shape[0]
        assert pred.dim() <= 2
        assert target.dim() <= 2
        assert target.dim() == 1 \
               or target.shape[1] == 1 \
               or target.shape[1] >= self.num_classes


        if pred.dim() == 2:
            pred = pred.argmax(dim=1)
        if pred.is_floating_point():
            pred = pred.long()


        if target.dim() == 2 and target.shape[1] >= self.num_classes:

            target = target[:, :self.num_classes]
            confmat = scatter_add(
                target.float(), pred, dim=0, dim_size=self.num_classes)
            self.confmat += confmat.T.long()
            return


        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze()


        target[target < 0 | target > self.num_classes] = self.num_classes


        super().update(pred, target)

    @classmethod
    def from_confusion_matrix(cls, confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        assert not confusion_matrix.is_floating_point()
        cm = cls(confusion_matrix.shape[0])
        cm.confmat = confusion_matrix.long()
        return cm

    @classmethod
    def from_histogram(cls, h):

        assert h.ndim == 2
        assert not h.is_floating_point()
        cm = cls(h.shape[1])
        cm(h.argmax(dim=1), h)
        return cm

    def iou(self, as_percent=True):

        TP_plus_FN = self.confmat.sum(dim=0)
        TP_plus_FP = self.confmat.sum(dim=1)
        TP = self.confmat.diag()
        union = TP_plus_FN + TP_plus_FP - TP
        iou = 1e-8 + TP / (union + 1e-8)
        existing_class_mask = union > 1e-3
        if as_percent:
            iou *= 100
        return iou, existing_class_mask

    def oa(self, as_percent=True):

        confusion_matrix = self.confmat
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.num_classes):
            for column in range(self.num_classes):
                all_values += confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        if as_percent:
            matrix_diagonal *= 100
        return float(matrix_diagonal) / all_values

    def miou(self, missing_as_one=False, as_percent=True):

        values, existing_classes_mask = self.iou(as_percent=as_percent)
        if existing_classes_mask.sum() == 0:
            return 0
        if missing_as_one:
            values[~existing_classes_mask] = 1
            existing_classes_mask[:] = True
        return values[existing_classes_mask].sum() / existing_classes_mask.sum()

    def macc(self, as_percent=True):

        re = 0
        label_presents = 0
        for i in range(self.num_classes):
            total_gt = self.confmat[i, :].sum()
            if total_gt:
                label_presents += 1
                re = re + self.confmat[i][i] / max(1, total_gt)
        if label_presents == 0:
            return 0
        if as_percent:
            re *= 100
        return re / label_presents

    def print_metrics(self, class_names=None):

        oa = self.oa()
        macc = self.macc()
        miou = self.miou()
        class_iou = self.iou()

        print(f'OA: {oa:0.2f}')
        print(f'mAcc: {macc:0.2f}')
        print(f'mIoU: {miou:0.2f}')

        class_names = class_names if class_names is not None \
            else [f'class-{i}' for i in range(self.num_classes)]
        for c, iou, seen in zip(class_names, class_iou[0], class_iou[1]):
            if not seen:
                print(f'  {c:<13}: not seen')
                continue
            print(f'  {c:<13}: {iou:0.2f}')

    def all_metrics(self, as_percent=True):

        metrics = SemanticMetricResults()
        metrics.oa = self.oa(as_percent=as_percent)
        metrics.macc = self.macc(as_percent=as_percent)
        metrics.miou = self.miou(as_percent=as_percent)
        iou, seen = self.iou(as_percent=as_percent)
        metrics.iou_per_class = iou
        metrics.seen_class = seen
        return metrics


def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=5)

    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().float().numpy()

    template_path = os.path.join(path2save, "{}.svg")

    cmn = cm.astype("float") / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names,
        yticklabels=ordered_names, annot_kws={"size": 20})

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")


    cmn = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names,
        yticklabels=ordered_names, annot_kws={"size": 20})

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")
