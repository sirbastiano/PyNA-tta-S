import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import math

from pynattas.builder.netBuilder import GenericNetwork

class GenericTrainer(pl.LightningModule):

    def __init__(self, parsed_layers, input_channels, input_height, input_width, num_classes, learning_rate=1e-3):
        """
        A PyTorch Lightning Module for a generic neural network with various metrics and logging.
        Args:
            parsed_layers (list): List of parsed layers for the network.
            input_channels (int): Number of input channels.
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
            num_classes (int): Number of output classes.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.
        Attributes:
            lr (float): Learning rate for the optimizer.
            model (GenericNetwork): The neural network model.
            loss_fn (nn.CrossEntropyLoss): Loss function.
            accuracy (torchmetrics.classification.BinaryAccuracy): Accuracy metric.
            f1_score (torchmetrics.classification.BinaryF1Score): F1 score metric.
            mcc (torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef): Matthews correlation coefficient metric.
            conf_matrix (torchmetrics.classification.BinaryConfusionMatrix): Confusion matrix for test data.
            conf_matrix_pred (torchmetrics.classification.BinaryConfusionMatrix): Confusion matrix for prediction data.
        Methods:
            forward(x):
                Forward pass through the network.
            training_step(batch, batch_idx):
                Training step for a single batch.
            validation_step(batch, batch_idx):
                Validation step for a single batch.
            test_step(batch, batch_idx):
                Test step for a single batch.
            on_test_end():
                Actions to perform at the end of the test phase, such as plotting and saving the confusion matrix.
            _common_step(batch, batch_idx):
                Common step for computing loss and scores.
            predict_step(batch, batch_idx):
                Prediction step for a single batch.
            configure_optimizers():
                Configures the optimizer for training.
        """
        super(GenericLightningNetwork, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_classes=num_classes,
        )

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1_score = torchmetrics.classification.BinaryF1Score()
        self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
        self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
        self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def on_test_end(self):
        fig_, ax_ = self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        # plt.show()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")   
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer


class GenericLightningNetwork_Custom(pl.LightningModule): # Old implementation with a different loss. Requires update
    def __init__(self, parsed_layers, model_parameters, input_channels,input_height,input_width, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork_Custom, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            model_parameters=model_parameters,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_classes=num_classes,
        )
        self.class_weights = None  # Initialize with a default value

        # Metrics
        self.loss_fn = ce_loss  # Use custom loss function
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1_score = torchmetrics.classification.BinaryF1Score()
        self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
        self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
        self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # Ensure the datamodule is attached and has class_weights
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'class_weights'):
            self.class_weights = self.trainer.datamodule.class_weights.to(self.device)
            print(f"GenericLightningNetwork class_weights set: {self.class_weights}")
        else:
            print("GenericLightningNetwork class_weights NOT set")

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def on_test_end(self):
        fig_, ax_ = self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        # plt.show()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        if self.class_weights is not None:
            loss = self.loss_fn(logits=scores, targets=y, weight=self.class_weights, use_hard_labels=True)
        else:
            loss = self.loss_fn(logits=scores, targets=y, use_hard_labels=True)

        loss = loss.mean()
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer


def ce_loss(logits, targets, weight=None, use_hard_labels=True, reduction="none"):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        weight: weights for loss if hard labels are used.
        use_hard_labels: If True, targets have [Batch size] shape with int values.
                         If False, the target is vector. Default to True.
    """
    if use_hard_labels:
        if weight is not None:
            return F.cross_entropy(
                logits, targets.long(), weight=weight, reduction=reduction
            )
        else:
            return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


################################################################# OD
# inspired by https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/main.py
# OD Head like YOLOv3 loss
"""ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
"""
ANCHORS = [ # obtained by normalizing over 416 the classic anchors
    (0.02403846153846154, 0.03125),
    (0.038461538461538464, 0.07211538461538461),
    (0.07932692307692307, 0.055288461538461536),
    (0.07211538461538461, 0.1466346153846154),
    (0.14903846153846154, 0.10817307692307693),
    (0.14182692307692307, 0.2860576923076923),
    (0.27884615384615385, 0.21634615384615385),
    (0.375, 0.47596153846153844),
    (0.8966346153846154, 0.7836538461538461)
    ] #"""
Tensor = torch.Tensor

class GenericOD_YOLOv3(pl.LightningModule):
    def __init__(
            self, 
            parsed_layers, 
            input_channels,
            input_height,
            input_width, 
            num_classes, 
            learning_rate=1e-3, 
            ):
        super().__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_classes=num_classes,
        )
        # Metrics
        #self.loss_fn = yolo_loss_fn()

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        self.train_loss = []
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        #print("\n0:", targets)
        #print("\n1:", targets[0])
        #print("\n2:", targets[0]['boxes'])
        #print("\n3:", targets[0]['boxes'][0])
        
        # Get the number of targets for each image in the batch
        tgt_len = torch.tensor([len(t['boxes']) for t in targets], device=self.device)
        #print("\n4:", tgt_len)
        
        # Create the target tensor
        max_tgt_len = max(tgt_len)
        #("\n5:", max_tgt_len)
        num_attributes = 6  # 4 for bbox, 1 for objectness, 1 for class label
        tgt = torch.zeros((len(targets), max_tgt_len, num_attributes), device=self.device)
        
        for i, t in enumerate(targets):
            num_objs = len(t['boxes'])
            tgt[i, :num_objs, :4] = t['boxes']
            tgt[i, :num_objs, 4] = 1  # objectness
            tgt[i, :num_objs, 5] = t['labels'].float()  # class label as float
        
        #print("\n6:", tgt)
        
        #print("Images:", images.shape, "Prediction:", preds.shape, "Target:",tgt.shape)
        loss, coord_loss, obj_loss, noobj_loss, class_loss = 0, 0, 0, 0, 0
        loss, coord_loss, obj_loss, noobj_loss, class_loss = yolo_loss_fn(preds, tgt, tgt_len, img_size=images.size(2))
        
        self.log_dict({
            'train_loss': loss, 
            'train_coord_loss': coord_loss,
            'train_obj_loss': obj_loss, 
            'train_noobj_loss': noobj_loss, 
            'train_class_loss': class_loss
            })

        self.train_loss.append(loss.cpu().detach().numpy())

        return loss
    
    def on_train_epoch_end(self):
        epoch_train_loss = np.mean(self.train_loss)
        print(f"Train Loss: {epoch_train_loss}")

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        
        # Get the number of targets for each image in the batch
        tgt_len = torch.tensor([len(t['boxes']) for t in targets], device=self.device)
      
        # Create the target tensor
        max_tgt_len = max(tgt_len)
        num_attributes = 6  # 4 for bbox, 1 for objectness, 1 for class label
        tgt = torch.zeros((len(targets), max_tgt_len, num_attributes), device=self.device)
        
        for i, t in enumerate(targets):
            num_objs = len(t['boxes'])
            tgt[i, :num_objs, :4] = t['boxes']
            tgt[i, :num_objs, 4] = 1  # objectness
            tgt[i, :num_objs, 5] = t['labels'].float()  # class label as float
        
        loss, coord_loss, obj_loss, noobj_loss, class_loss = 0, 0, 0, 0, 0
        loss, coord_loss, obj_loss, noobj_loss, class_loss = yolo_loss_fn(preds, tgt, tgt_len, img_size=images.size(2))
        
        self.log_dict({
            'val_loss': loss, 
            'val_coord_loss': coord_loss,
            'val_obj_loss': obj_loss, 
            'val_noobj_loss': noobj_loss, 
            'val_class_loss': class_loss
            })

        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        
        # Get the number of targets for each image in the batch
        tgt_len = torch.tensor([len(t['boxes']) for t in targets], device=self.device)
        
        # Create the target tensor
        max_tgt_len = max(tgt_len)
        num_attributes = 6  # 4 for bbox, 1 for objectness, 1 for class label
        tgt = torch.zeros((len(targets), max_tgt_len, num_attributes), device=self.device)
        
        for i, t in enumerate(targets):
            num_objs = len(t['boxes'])
            tgt[i, :num_objs, :4] = t['boxes']
            tgt[i, :num_objs, 4] = 1  # objectness
            tgt[i, :num_objs, 5] = t['labels'].float()  # class label as float
        
        loss, coord_loss, obj_loss, noobj_loss, class_loss = 0, 0, 0, 0, 0
        loss, coord_loss, obj_loss, noobj_loss, class_loss = yolo_loss_fn(preds, tgt, tgt_len, img_size=images.size(2))
        
        self.log_dict({
            'test_loss': loss, 
            'test_coord_loss': coord_loss,
            'test_obj_loss': obj_loss, 
            'test_noobj_loss': noobj_loss, 
            'test_class_loss': class_loss
            })

        return loss

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer
    

# My version
# Define the loss function
def yolo_loss_fn(preds: Tensor, tgt: Tensor, tgt_len: Tensor, img_size: int, average=True):
    """Calculate the loss function given the predictions, the targets, the length of each target and the image size."""
    #print(f"Preds shape: {preds.shape}")
    #print(f"Tgt shape: {tgt.shape}")
    #print(f"Tgt_len: {tgt_len}")
    #print(f"Img_size: {img_size}")
    if torch.isnan(preds).any():
        print("NaNs found in preds before processing")

    # generate the no-objectness mask. mask_noobj has size of [B, N_PRED]
    mask_noobj = noobj_mask_fn(preds, tgt)
    #print("The number of non_zeros in mask_noobj is:", torch.count_nonzero(mask_noobj))
    #print(f"Mask_noobj shape: {mask_noobj.shape}")
    #print(f"mask_noobj - {mask_noobj}")

    tgt_t_1d, idx_pred_obj = pre_process_targets(tgt, tgt_len, img_size)
    #print(f"Tgt_t_1d shape: {tgt_t_1d.shape}")
    #print("tgt_t_1d - ", tgt_t_1d)
    #print("idx_pred_obj - ", idx_pred_obj)
    #print(f"Idx_pred_obj shape: {idx_pred_obj.shape}")

    mask_noobj = noobj_mask_filter(mask_noobj, idx_pred_obj)
    #print("The number of non_zeros in mask_noobj after filter is:", torch.count_nonzero(mask_noobj))
    #print(f"Mask_noobj after filter shape: {mask_noobj.shape}")

    # calculate the no-objectness loss
    pred_conf_logit = preds[..., 4]
    #print(f"pred_conf_logit before: {pred_conf_logit}")
    tgt_zero = torch.zeros(pred_conf_logit.size(), device=pred_conf_logit.device)
    pred_conf_logit = pred_conf_logit - (1 - mask_noobj) * 1e6
    noobj_loss = F.binary_cross_entropy_with_logits(pred_conf_logit, tgt_zero, reduction='mean')
    #print(f"Noobj_loss: {noobj_loss}")

    # select the predictions corresponding to the targets
    n_batch, n_pred, _ = preds.size()
    preds_1d = preds.view(n_batch * n_pred, -1)
    preds_obj = preds_1d.index_select(0, idx_pred_obj)
    #print(f"Preds_obj shape: {preds_obj.shape}")

    # calculate the coordinate loss
    coord_loss = F.mse_loss(preds_obj[..., :4], tgt_t_1d[..., :4], reduction='mean')
    #print(f"Coord_loss: {coord_loss}")

    # calculate the objectness loss
    pred_conf_obj_logit = preds_obj[..., 4]
    tgt_one = torch.ones(pred_conf_obj_logit.size(), device=pred_conf_obj_logit.device)
    obj_loss = F.binary_cross_entropy_with_logits(pred_conf_obj_logit, tgt_one, reduction='mean')
    #print(f"Obj_loss: {obj_loss}")

    # Convert class indices to one-hot encoding
    num_classes = preds.shape[-1] - 5
    tgt_t_1d_classes = F.one_hot(tgt_t_1d[..., 5].long(), num_classes).float()
    #print(f"Tgt_t_1d_classes shape: {tgt_t_1d_classes.shape}")

    # calculate the classification loss
    class_loss = F.binary_cross_entropy_with_logits(preds_obj[..., 5:], tgt_t_1d_classes, reduction='mean')
    #print(f"Class_loss: {class_loss}")

    # total loss
    noobj_coeff = 0.2
    coord_coeff = 5
    total_loss = noobj_loss * noobj_coeff + obj_loss + class_loss + coord_loss * coord_coeff

    if average:
        total_loss = total_loss / n_batch
    
    #print(f"Total_loss: {total_loss}")
    return total_loss, coord_loss, obj_loss, noobj_loss, class_loss

def noobj_mask_fn(pred: Tensor, target: Tensor):
    ignore_threshold = 0.5
    num_batch, num_pred, num_attrib = pred.size()
    assert num_batch == target.size(0)
    
    # Calculate IoU between decoded predictions and targets
    ious = iou_batch(pred[..., :4], target[..., :4], center=True) #in cxcywh format
    #print("IoUs shape in noobj_mask_fn:", ious.shape)
    #print(f"IoU min: {ious.min().item()}, max: {ious.max().item()}, mean: {ious.mean().item()}, std: {ious.std().item()}")

    # for each pred bbox, find the target box which overlaps with it (without zero centered) most, and the iou value.
    max_ious, max_ious_idx = torch.max(ious, dim=2)
    #print("Max_IoUs shape in noobj_mask_fn:", max_ious.shape)
    noobj_indicator = torch.where((max_ious - ignore_threshold) > 0, torch.zeros_like(max_ious), torch.ones_like(max_ious))
    
    return noobj_indicator

def noobj_mask_filter(mask_noobj: Tensor, idx_obj_1d: Tensor):
    n_batch, n_pred = mask_noobj.size()
    mask_noobj = mask_noobj.view(-1)
    filter_ = torch.zeros(mask_noobj.size(), device=mask_noobj.device)
    mask_noobj.scatter_(0, idx_obj_1d, filter_)
    mask_noobj = mask_noobj.view(n_batch, -1)
    return mask_noobj

def pre_process_targets(tgt: Tensor, tgt_len, img_size):
    epsilon = 1e-9

    # Initializes the anchor boxes and creates a tensor of shape [1, n_anchor, 4] where n_anchor is the number of anchors. 
    # Each anchor box has zero-centered coordinates (cx, cy, w, h).
    wh_anchor = torch.tensor(ANCHORS).to(tgt.device).float()
    n_anchor = wh_anchor.size(0)
    xy_anchor = torch.zeros((n_anchor, 2), device=tgt.device)
    bbox_anchor = torch.cat((xy_anchor, wh_anchor), dim=1)
    bbox_anchor.unsqueeze_(0)

    #print("bbox_anchor:", bbox_anchor)
    #bbox_anchor /= img_size + epsilon
    #print("bbox_anchor/img_size:", bbox_anchor)

    # Computes the Intersection Over Union (IOU) between each anchor and target box, then finds the anchor with the maximum 
    # IOU for each target.
    iou_anchor_tgt = iou_batch(bbox_anchor, tgt[..., :4], zero_center=True)
    #print("iou_anchor_tgt:", iou_anchor_tgt)
    _, idx_anchor = torch.max(iou_anchor_tgt, dim=1)
    #print("idx_anchor:", idx_anchor)

    # Calculates the grid cell coordinates (grid_x, grid_y) where each target falls, based on the stride of the respective scale (8, 16, or 32).
    # Determines the corresponding prediction index for each target, taking into account the scale and the position within the grid.
    strides_selection = [8, 16, 32]
    scale = idx_anchor // 3
    #print("scale:", scale)
    idx_anchor_by_scale = idx_anchor - scale * 3
    #print("idx_anchor_by_scale:", idx_anchor_by_scale)
    stride = 8 * 2 ** scale
    #print("stride:", stride)
    #print("tgt[..., 0]:", tgt[..., 0], "tgt[..., 1]:", tgt[..., 1])
    grid_x = ((tgt[..., 0]*img_size) // stride.float()).long()
    grid_y = ((tgt[..., 1]*img_size) // stride.float()).long()
    #print("grid_x:", grid_x, "grid_y:", grid_y)
    n_grid = img_size // stride
    #print("n_grid:", n_grid)
    large_scale_mask = (scale <= 1).long()
    med_scale_mask = (scale <= 0).long()
    idx_obj = \
        large_scale_mask * (img_size // strides_selection[2]) ** 2 * 3 + \
        med_scale_mask * (img_size // strides_selection[1]) ** 2 * 3 + \
        n_grid ** 2 * idx_anchor_by_scale + n_grid * grid_y + grid_x
    #print("idx_obj:", idx_obj)

    # Calculate Local Coordinates (tx, ty, tw, th):
    t_x = ((tgt[..., 0]*img_size) / stride.float() - grid_x.float()).clamp(epsilon, 1 - epsilon)
    t_x = torch.log(t_x / (1. - t_x))   #inverse of sigmoid
    t_y = ((tgt[..., 1]*img_size) / stride.float() - grid_y.float()).clamp(epsilon, 1 - epsilon)
    t_y = torch.log(t_y / (1. - t_y))   # inverse of sigmoid

    w_anchor = wh_anchor[..., 0] #/ (img_size + epsilon)
    h_anchor = wh_anchor[..., 1] #/ (img_size + epsilon)
    w_anchor = torch.index_select(w_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    h_anchor = torch.index_select(h_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    t_w = torch.log((tgt[..., 2] / w_anchor).clamp(min=epsilon))
    t_h = torch.log((tgt[..., 3] / h_anchor).clamp(min=epsilon))

    # Create the Processed Target Tensor:
    # Updates the target tensor tgt_t with the local coordinates.
    tgt_t = tgt.clone().detach()

    tgt_t[..., 0] = t_x
    tgt_t[..., 1] = t_y
    tgt_t[..., 2] = t_w
    tgt_t[..., 3] = t_h

    # Aggregate Processed Targets and Indices:
    n_batch = tgt.size(0)
    n_pred = sum([(img_size // s) ** 2 for s in strides_selection]) * 3

    idx_obj_1d = []
    tgt_t_flat = []

    for i_batch in range(n_batch):
        v = idx_obj[i_batch]
        t = tgt_t[i_batch]
        l = tgt_len[i_batch]
        idx_obj_1d.append(v[:l] + i_batch * n_pred)
        tgt_t_flat.append(t[:l])

    idx_obj_1d = torch.cat(idx_obj_1d)
    tgt_t_flat = torch.cat(tgt_t_flat)

    return tgt_t_flat, idx_obj_1d

def iou_batch(bboxes1: Tensor, bboxes2: Tensor, center=False, zero_center=False):
    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]

    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]

    area1 = w1 * h1
    area2 = w2 * h2

    epsilon = 1e-9

    if zero_center:
        w1.unsqueeze_(2)
        w2.unsqueeze_(1)
        h1.unsqueeze_(2)
        h2.unsqueeze_(1)
        w_intersect = torch.min(w1, w2).clamp(min=0)
        h_intersect = torch.min(h1, h2).clamp(min=0)
    else:
        if center:
            x1 = x1 - (w1 / 2)
            y1 = y1 - (h1 / 2)
            x2 = x2 - (w2 / 2)
            y2 = y2 - (h2 / 2)
        right1 = (x1 + w1).unsqueeze(2)
        right2 = (x2 + w2).unsqueeze(1)
        top1 = (y1 + h1).unsqueeze(2)
        top2 = (y2 + h2).unsqueeze(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        bottom1 = y1.unsqueeze(2)
        bottom2 = y2.unsqueeze(1)
        w_intersect = (torch.min(right1, right2) - torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) - torch.max(bottom1, bottom2)).clamp(min=0)

    area_intersect = h_intersect * w_intersect
    iou_ = area_intersect / (area1.unsqueeze(2) + area2.unsqueeze(1) - area_intersect + epsilon)
    #print("IoU:", iou_)

    return iou_

#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
# OD Head like YOLOv3 with a focus on Small Objects
"""
    ANCHORS_SmallObjects = [(10, 25), (15, 15), (25, 10), (25, 50), (35, 35), (50, 25)]
    ANCHORS_SmallObjects = [(19, 14), (22, 29), (27, 44), (30, 17), (45, 21), (73, 25)]
"""
'''
ANCHORS_SmallObjects = [# obtained by normalizing over 512
    (0.01953125, 0.04882812),
    (0.02929688, 0.02929688),
    (0.04882812, 0.01953125),
    (0.04882812, 0.09765625),
    (0.06835938, 0.06835938),
    (0.09765625, 0.04882812)
    ]
'''

#Anchor Box 1: 18.81 x 14.03 with aspect ratio 1.41
#Anchor Box 2: 21.75 x 28.86 with aspect ratio 0.77
#Anchor Box 4: 29.78 x 16.55 with aspect ratio 1.89
#Anchor Box 3: 27.00 x 44.23 with aspect ratio 0.62
#Anchor Box 5: 45.17 x 21.15 with aspect ratio 2.25
#Anchor Box 6: 72.67 x 24.50 with aspect ratio 3.15

ANCHORS_SmallObjects = [# obtained by normalizing over 512
    (0.03673828, 0.02740234),
    (0.04248047, 0.05636719),
    (0.05816406, 0.03232422),
    (0.05273438, 0.08638672),
    (0.08822266, 0.04130859),
    (0.14193359, 0.04785156)
]
Tensor = torch.Tensor

class GenericOD_YOLOv3_SmallObjects(pl.LightningModule):
    def __init__(
            self, 
            parsed_layers, 
            input_channels,
            input_height,
            input_width, 
            num_classes, 
            learning_rate=1e-3, 
            ):
        super().__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_classes=num_classes,
        )
        # Metrics
        #self.loss_fn = yolo_loss_fn()

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        self.train_loss_vec = []
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        #print("\n0:", targets)
        #print("\n1:", targets[0])
        #print("\n2:", targets[0]['boxes'])
        #print("\n3:", targets[0]['boxes'][0])
        
        # Get the number of targets for each image in the batch
        tgt_len = torch.tensor([len(t['boxes']) for t in targets], device=self.device)
        #print("\n4:", tgt_len)
        
        # Create the target tensor
        max_tgt_len = max(tgt_len)
        #("\n5:", max_tgt_len)
        num_attributes = 6  # 4 for bbox, 1 for objectness, 1 for class label
        tgt = torch.zeros((len(targets), max_tgt_len, num_attributes), device=self.device)
        
        for i, t in enumerate(targets):
            num_objs = len(t['boxes'])
            if num_objs == 0:
                #print(f"No objects in this training image: batch_idx = {batch_idx}, index_in_batch = {i}. Continuing:")
                continue
            tgt[i, :num_objs, :4] = t['boxes']
            tgt[i, :num_objs, 4] = 1  # objectness
            tgt[i, :num_objs, 5] = t['labels'].float()  # class label as float
        
        #print("\n6:", tgt)
        
        #print("Images:", images.shape, "Prediction:", preds.shape, "Target:",tgt.shape)
        #loss, coord_loss, obj_loss, noobj_loss, class_loss = 0, 0, 0, 0, 0
        loss, coord_loss, obj_loss, noobj_loss, class_loss = yolo_loss_fn_SO(preds, tgt, tgt_len, img_size=images.size(2))
        
        self.log_dict({
            'train_loss': loss, 
            'train_coord_loss': coord_loss,
            'train_obj_loss': obj_loss, 
            'train_noobj_loss': noobj_loss, 
            'train_class_loss': class_loss
            })

        self.train_loss_vec.append(loss.cpu().detach().numpy())

        return loss
    
    def on_train_epoch_end(self):
        epoch_train_loss = np.mean(self.train_loss_vec)
        print(f"Train Loss: {epoch_train_loss}")

    def on_validation_epoch_start(self):
        print("Validating...")
        self.val_loss_vec = []

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        
        # Get the number of targets for each image in the batch
        tgt_len = torch.tensor([len(t['boxes']) for t in targets], device=self.device)
      
        # Create the target tensor
        max_tgt_len = max(tgt_len)
        num_attributes = 6  # 4 for bbox, 1 for objectness, 1 for class label
        tgt = torch.zeros((len(targets), max_tgt_len, num_attributes), device=self.device)
        
        for i, t in enumerate(targets):
            num_objs = len(t['boxes'])
            if num_objs == 0:
                #print(f"No objects in this validation image: batch_idx = {batch_idx}, index_in_batch = {i}. Continuing:")
                continue
            tgt[i, :num_objs, :4] = t['boxes']
            tgt[i, :num_objs, 4] = 1  # objectness
            tgt[i, :num_objs, 5] = t['labels'].float()  # class label as float
        
        #loss, coord_loss, obj_loss, noobj_loss, class_loss = 0, 0, 0, 0, 0
        loss, coord_loss, obj_loss, noobj_loss, class_loss = yolo_loss_fn_SO(preds, tgt, tgt_len, img_size=images.size(2))
        
        self.log_dict({
            'val_loss': loss, 
            'val_coord_loss': coord_loss,
            'val_obj_loss': obj_loss, 
            'val_noobj_loss': noobj_loss, 
            'val_class_loss': class_loss
            })
        
        self.val_loss_vec.append(loss.cpu().detach().numpy())

        return loss
    
    def on_validation_epoch_end(self):
        epoch_val_loss = np.mean(self.val_loss_vec)
        print(f"Validation Loss: {epoch_val_loss}")

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        
        # Get the number of targets for each image in the batch
        tgt_len = torch.tensor([len(t['boxes']) for t in targets], device=self.device)
        
        # Create the target tensor
        max_tgt_len = max(tgt_len)
        num_attributes = 6  # 4 for bbox, 1 for objectness, 1 for class label
        tgt = torch.zeros((len(targets), max_tgt_len, num_attributes), device=self.device)
        
        for i, t in enumerate(targets):
            num_objs = len(t['boxes'])
            if num_objs == 0:
                #print(f"No objects in this testing image: batch_idx = {batch_idx}, index_in_batch = {i}. Continuing:")
                continue
            tgt[i, :num_objs, :4] = t['boxes']
            tgt[i, :num_objs, 4] = 1  # objectness
            tgt[i, :num_objs, 5] = t['labels'].float()  # class label as float
        
        #loss, coord_loss, obj_loss, noobj_loss, class_loss = 0, 0, 0, 0, 0
        loss, coord_loss, obj_loss, noobj_loss, class_loss = yolo_loss_fn_SO(preds, tgt, tgt_len, img_size=images.size(2))

        test_SIoU = 1 - coord_loss # mean SIoU
        
        self.log_dict({
            'test_loss': loss, 
            'test_coord_loss': coord_loss,
            'test_obj_loss': obj_loss, 
            'test_noobj_loss': noobj_loss, 
            'test_class_loss': class_loss,
            'test_SIoU': test_SIoU
            })

        return loss

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer
    

# My version
# Define the loss function
def yolo_loss_fn_SO(preds: Tensor, tgt: Tensor, tgt_len: Tensor, img_size: int, average=True):
    """Calculate the loss function given the predictions, the targets, the length of each target and the image size."""

    #if torch.isnan(preds).any():
    #    print("NaNs found in preds before processing")
    assert not torch.isnan(preds).any(), "NaNs found in preds before loss"

    # generate the no-objectness mask. mask_noobj has size of [B, N_PRED]
    mask_noobj = noobj_mask_fn_SO(preds, tgt)
    tgt_t_1d, idx_pred_obj = pre_process_targets_SO(tgt, tgt_len, img_size)
    mask_noobj = noobj_mask_filter_SO(mask_noobj, idx_pred_obj)

    # calculate the no-objectness loss
    pred_conf = preds[..., 4]
    pred_conf_logit = torch.log(1e-6 + pred_conf / (1 - pred_conf + 1e-6))
    tgt_zero = torch.zeros(pred_conf_logit.size(), device=pred_conf_logit.device)
    pred_conf_logit = pred_conf_logit - (1 - mask_noobj) * 1e6
    noobj_loss = F.binary_cross_entropy_with_logits(pred_conf_logit, tgt_zero, reduction='mean')
    #print(f"Noobj_loss: {noobj_loss}")

    # select the predictions corresponding to the targets
    n_batch, n_pred, _ = preds.size()
    preds_1d = preds.view(n_batch * n_pred, -1)
    preds_obj = preds_1d.index_select(0, idx_pred_obj)

    ### MSE loss (default)
    #coord_loss = F.huber_loss(preds_obj[..., :4], tgt_t_1d[..., :4], reduction='mean')
    tgt_flat = flatten_tgt(tgt, tgt_len)
    preds_scaled = preds_obj[..., :4] * img_size
    tgt_scaled = tgt_flat[..., :4] * img_size
    #coord_loss = F.mse_loss(preds_scaled, tgt_scaled, reduction='sum')
    coord_loss = siou_loss(preds_scaled, tgt_scaled, gamma=0.5, kappa=8, eps=1e-6, reduction='mean')
    #print(f"SIoU mean loss: {coord_loss}")
    #coord_loss = 1 / ((torch.sum(iou_batch_SO(preds_scaled, tgt_scaled, center=True))) + 1e-6)
    #print(f"Coord_loss: {coord_loss}")

    # calculate the objectness loss
    pred_conf_obj_logit = preds_obj[..., 4]
    tgt_one = torch.ones(pred_conf_obj_logit.size(), device=pred_conf_obj_logit.device)
    obj_loss = F.binary_cross_entropy_with_logits(pred_conf_obj_logit, tgt_one, reduction='mean')
    #print(f"Obj_loss: {obj_loss}")

    # Convert class indices to one-hot encoding
    num_classes = preds.shape[-1] - 5
    if num_classes == 1:
        # Calculate the classification loss for a single class
        class_loss = 0
    else:
        tgt_t_1d_classes = F.one_hot(tgt_t_1d[..., 5].long(), num_classes).float()

        # calculate the classification loss
        class_loss = F.binary_cross_entropy_with_logits(preds_obj[..., 5:], tgt_t_1d_classes, reduction='sum')
    #print(f"Class_loss: {class_loss}")

    # total loss
    noobj_coeff = 0.2 # 0.2
    coord_coeff = 10 # 5
    obj_loss_coeff = 1
    #total_loss = coord_coeff * coord_loss
    total_loss = noobj_loss * noobj_coeff + obj_loss * obj_loss_coeff + class_loss + coord_loss * coord_coeff
    #total_loss = noobj_loss * noobj_coeff + obj_loss + class_loss + coord_loss * 100
    #print(f"Total_loss: {total_loss}")

    #print(f"Total_loss: {total_loss}")

    if average:
        total_loss = total_loss / n_batch
    
    #print(f"Total_loss: {total_loss}")
    return total_loss, coord_loss, obj_loss, noobj_loss, class_loss

def noobj_mask_fn_SO(pred: Tensor, target: Tensor, img_size=512):
    ignore_threshold = 0.5
    num_batch, num_pred, num_attrib = pred.size()
    assert num_batch == target.size(0)
    
    ## DEFAULT: Calculate IoU between decoded predictions and targets
    ious = iou_batch_SO(pred[..., :4], target[..., :4], center=True) #in cxcywh format
    # Check if there are any targets
    if target.size(1) == 0:
        # No targets; all predictions are considered as no-object
        noobj_indicator = torch.ones((num_batch, num_pred), device=pred.device)
    else:
        # Calculate IoU between decoded predictions and targets
        ious = iou_batch_SO(pred[..., :4], target[..., :4], center=True)  # in cxcywh format

        # For each prediction, find the target box with the highest IoU
        max_ious, max_ious_idx = torch.max(ious, dim=2)
        noobj_indicator = torch.where(
            (max_ious - ignore_threshold) > 0,
            torch.zeros_like(max_ious),
            torch.ones_like(max_ious)
        )

    ## Attempt to use SIoU instead of IoU
    #preds_scaled = pred[..., :4] * img_size
    #tgt_scaled = target[..., :4] * img_size
    ## SIoU as implimented expects bounding boxes in (cx, cy, w, h) format, scaled to image size.
    #ious = siou(preds_scaled, tgt_scaled)

    return noobj_indicator

def noobj_mask_filter_SO(mask_noobj: Tensor, idx_obj_1d: Tensor):
    n_batch, n_pred = mask_noobj.size()
    mask_noobj = mask_noobj.view(-1)
    filter_ = torch.zeros(mask_noobj.size(), device=mask_noobj.device)
    mask_noobj.scatter_(0, idx_obj_1d, filter_)
    mask_noobj = mask_noobj.view(n_batch, -1)
    return mask_noobj

def pre_process_targets_SO(tgt: Tensor, tgt_len, img_size):
    epsilon = 1e-9

    # Initializes the anchor boxes and creates a tensor of shape [1, n_anchor, 4] where n_anchor is the number of anchors. 
    # Each anchor box has zero-centered coordinates (cx, cy, w, h).
    wh_anchor = torch.tensor(ANCHORS_SmallObjects).to(tgt.device).float()
    n_anchor = wh_anchor.size(0)
    xy_anchor = torch.zeros((n_anchor, 2), device=tgt.device)
    bbox_anchor = torch.cat((xy_anchor, wh_anchor), dim=1)
    bbox_anchor.unsqueeze_(0)

    #print("bbox_anchor:", bbox_anchor)
    #bbox_anchor /= img_size + epsilon
    #print("bbox_anchor/img_size:", bbox_anchor)

    ## DEFAULT: Computes the Intersection Over Union (IOU) between each anchor and target box, then finds the anchor with the maximum 
    ## IOU for each target.
    iou_anchor_tgt = iou_batch_SO(bbox_anchor, tgt[..., :4], zero_center=True)

    ## Attempt to use SIoU instead of IoU
    #scaled_anchor_bbox = bbox_anchor * img_size
    #tgt_scaled = tgt[..., :4] * img_size
    ## SIoU as implemented expects bounding boxes in (cx, cy, w, h) format, scaled to image size.
    #iou_anchor_tgt = siou(scaled_anchor_bbox, tgt_scaled)

    _, idx_anchor = torch.max(iou_anchor_tgt, dim=1)

    # Calculates the grid cell coordinates (grid_x, grid_y) where each target falls, based on the stride of the respective scale (8, 16, or 32).
    # Determines the corresponding prediction index for each target, taking into account the scale and the position within the grid.
    strides_selection = [8, 16]
    scale = idx_anchor // 3
    idx_anchor_by_scale = idx_anchor - scale * 3
    stride = 8 * 2 ** scale
    grid_x = ((tgt[..., 0]*img_size) // stride.float()).long()
    grid_y = ((tgt[..., 1]*img_size) // stride.float()).long()
    n_grid = img_size // stride
    med_scale_mask = (scale <= 0).long()
    idx_obj = \
        med_scale_mask * (img_size // strides_selection[1]) ** 2 * 3 + \
        n_grid ** 2 * idx_anchor_by_scale + n_grid * grid_y + grid_x

    # Calculate Local Coordinates (tx, ty, tw, th):
    t_x = ((tgt[..., 0]*img_size) / stride.float() - grid_x.float()).clamp(epsilon, 1 - epsilon)
    t_x = torch.log(t_x / (1. - t_x))   #inverse of sigmoid
    t_y = ((tgt[..., 1]*img_size) / stride.float() - grid_y.float()).clamp(epsilon, 1 - epsilon)
    t_y = torch.log(t_y / (1. - t_y))   # inverse of sigmoid

    w_anchor = wh_anchor[..., 0] #/ (img_size + epsilon)
    h_anchor = wh_anchor[..., 1] #/ (img_size + epsilon)
    w_anchor = torch.index_select(w_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    h_anchor = torch.index_select(h_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    t_w = torch.log((tgt[..., 2] / w_anchor).clamp(min=epsilon))
    t_h = torch.log((tgt[..., 3] / h_anchor).clamp(min=epsilon))

    # Create the Processed Target Tensor:
    # Updates the target tensor tgt_t with the local coordinates.
    tgt_t = tgt.clone().detach()

    tgt_t[..., 0] = t_x
    tgt_t[..., 1] = t_y
    tgt_t[..., 2] = t_w
    tgt_t[..., 3] = t_h

    # Aggregate Processed Targets and Indices:
    n_batch = tgt.size(0)
    n_pred = sum([(img_size // s) ** 2 for s in strides_selection]) * 3

    idx_obj_1d = []
    tgt_t_flat = []

    for i_batch in range(n_batch):
        v = idx_obj[i_batch]
        t = tgt_t[i_batch]
        l = tgt_len[i_batch]
        idx_obj_1d.append(v[:l] + i_batch * n_pred)
        tgt_t_flat.append(t[:l])

    idx_obj_1d = torch.cat(idx_obj_1d)
    tgt_t_flat = torch.cat(tgt_t_flat)

    return tgt_t_flat, idx_obj_1d

def iou_batch_SO(bboxes1: Tensor, bboxes2: Tensor, center=False, zero_center=False):
    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]

    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]

    area1 = w1 * h1
    area2 = w2 * h2

    epsilon = 1e-9

    if zero_center:
        w1.unsqueeze_(2)
        w2.unsqueeze_(1)
        h1.unsqueeze_(2)
        h2.unsqueeze_(1)
        w_intersect = torch.min(w1, w2).clamp(min=0)
        h_intersect = torch.min(h1, h2).clamp(min=0)
    else:
        if center:
            x1 = x1 - (w1 / 2)
            y1 = y1 - (h1 / 2)
            x2 = x2 - (w2 / 2)
            y2 = y2 - (h2 / 2)
        right1 = (x1 + w1).unsqueeze(2)
        right2 = (x2 + w2).unsqueeze(1)
        top1 = (y1 + h1).unsqueeze(2)
        top2 = (y2 + h2).unsqueeze(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        bottom1 = y1.unsqueeze(2)
        bottom2 = y2.unsqueeze(1)
        w_intersect = (torch.min(right1, right2) - torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) - torch.max(bottom1, bottom2)).clamp(min=0)

    area_intersect = h_intersect * w_intersect
    iou_ = area_intersect / (area1.unsqueeze(2) + area2.unsqueeze(1) - area_intersect + epsilon)
    #print("IoU:", iou_)

    return iou_

def focal_loss(preds, targets, alpha=0.25, gamma=2.0):
    """
    Compute the focal loss between `preds` and the ground truth `targets`.
    """
    bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    pt = torch.exp(-bce_loss)  # prevents nans when probability 0
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.sum()

def calculate_siou(pred_boxes, target_boxes, gamma=0.5, kappa=8, eps=1e-6):
    """
    Calculate the Scaled Intersection over Union (SIoU) between two bounding boxes.

    Parameters:
    -----------
    pred_boxes: Tensor of shape [N, 4]
        Predicted bounding boxes in (x, y, w, h) format, where (x, y) is the top-left corner.
    target_boxes: Tensor of shape [N, 4]
        Target bounding boxes in (x, y, w, h) format, where (x, y) is the top-left corner.
    gamma: float
        Scaling factor controlling the impact of the size of the bounding boxes.
    kappa: float
        Parameter related to the scale of the bounding boxes.
    eps: float
        Small value to avoid division by zero.

    Returns:
    --------
    Tensor
        The Scaled Intersection over Union (SIoU) between the two bounding boxes.
    """

    assert not torch.isnan(pred_boxes).any(), "pred_boxes contains NaNs"
    assert not torch.isnan(target_boxes).any(), "target_boxes contains NaNs"
    
    x1, y1, w1, h1 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x2, y2, w2, h2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_xmin = torch.max(x1, x2)
    inter_ymin = torch.max(y1, y2)
    inter_xmax = torch.min(x1_max, x2_max)
    inter_ymax = torch.min(y1_max, y2_max)

    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    # IoU calculation
    #where_iou = (inter_area > 0) * (union_area > 0)
    iou = torch.ones_like(x1) * eps
    iou = inter_area / (union_area + eps)
    iou = torch.nan_to_num(iou, nan=0.0, posinf=1.0, neginf=0.0)
    #assert not torch.isnan(iou).any(), "NaNs appeared after IoU calculation"
    if torch.isnan(iou).any():
        print("NaNs appeared after IoU calculation")
        print("inter_area:", inter_area)
        print("union_area:", union_area)
        print("iou:", iou)

    # Size factor calculation
    size_factor = torch.sqrt(area1 + area2)
    size_factor = torch.nan_to_num(size_factor, nan=0.0, posinf=1.0, neginf=0.0)
    #print("area1:", area1, "- area2:", area2, "- area1+area2:", area1+area2, "- size_factor:", size_factor)
    if torch.isnan(size_factor).any():
        print("NaNs appeared after size_factor calculation")
        print("area1:", area1)
        print("area2:", area2)
        print("size_factor:", size_factor)

    p = 1 - gamma * torch.exp(- size_factor / (math.sqrt(2)*kappa))
    assert not torch.isnan(p).any(), "NaNs appeared after p calculation"

    # SIoU calculation
    siou = torch.pow(iou + eps, p) # Maybe add eps to IoU to prevent pow(0, p) when IoU is zero
    assert not torch.isnan(siou).any(), "NaNs appeared after SIoU calculation"

    #print("IoU:", iou, "- SIoU:", siou)

    return siou

def siou_loss(pred_boxes, target_boxes, gamma=0.5, kappa=8, eps=1e-7, reduction='mean'):
    """
    Compute the SIoU loss between predicted and target bounding boxes.

    Parameters:
    -----------
    pred_boxes: Tensor of shape [N, 4]
        Predicted bounding boxes in (cx, cy, w, h) format, scaled to image size.
    target_boxes: Tensor of shape [N, 4]
        Target bounding boxes in (cx, cy, w, h) format, scaled to image size.
    gamma: float
        Scaling factor controlling the impact of the size of the bounding boxes.
    kappa: float
        Parameter related to the scale of the bounding boxes.
    eps: float
        Small value to avoid division by zero.
    reduction: str
        Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'

    Returns:
    --------
    Tensor
        SIoU loss value.
    """
    # Convert boxes from (cx, cy, w, h) to (x, y, w, h)
    pred_boxes_xywh = cxcywh_to_xywh(pred_boxes)
    target_boxes_xywh = cxcywh_to_xywh(target_boxes)

    # Calculate SIoU
    siou_values = calculate_siou(pred_boxes_xywh, target_boxes_xywh, gamma, kappa, eps)

    # SIoU loss is defined as 1 - SIoU
    siou_losses = 1 - siou_values

    #print("SIoU Losses:", siou_losses)
    #print("SIoU Loss:", siou_losses.mean())

    if reduction == 'mean':
        return torch.mean(siou_losses)
    elif reduction == 'sum':
        return torch.sum(siou_losses)
    elif reduction == 'none':
        return siou_losses
    else:
        raise ValueError(f"Invalid reduction mode '{reduction}'. Choose 'mean', 'sum', or 'none'.")


def cxcywh_to_xywh(bboxes):
    """
    Convert bounding boxes from (cx, cy, w, h) to (x, y, w, h) format.
    """
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    x = bboxes[..., 0] - w / 2
    y = bboxes[..., 1] - h / 2
    
    return torch.stack([x, y, w, h], dim=1)


def cxcywh_to_x1y1x2y2_batch(bboxes: Tensor):
    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return torch.stack((x1, y1, x2, y2), dim=-1)

def scale_boxes(bboxes, img_size):
    width, height = img_size

    x1 = bboxes[..., 0]
    y1 = bboxes[..., 1]
    x2 = bboxes[..., 2]
    y2 = bboxes[..., 3]

    x1 *= width
    y1 *= height 
    x2 *= width 
    y2 *= height 
    
    return torch.stack((x1, y1, x2, y2), dim=-1)

'''
def transform_predictions(prediction_raw, anchors, grid_size, stride, img_size):
    # Similar to what was in the model's head, but adapted for the loss function
    num_batch = prediction_raw.size(0)
    num_anchors = anchors.size(0)

    # Calculate offsets for each grid
    grid_tensor = torch.arange(grid_size, dtype=torch.float, device=prediction_raw.device)
    grid_x = grid_tensor.repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
    grid_y = grid_tensor.repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])

    anchor_w = anchors[:, 0:1].view((1, num_anchors, 1, 1))
    anchor_h = anchors[:, 1:2].view((1, num_anchors, 1, 1))

    # Apply transformations
    x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * stride / img_size  # Center x
    y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * stride / img_size  # Center y
    w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w / img_size  # Width
    h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h / img_size  # Height

    bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4)
    return bbox_pred.view(num_batch, -1, 4)
'''

def flatten_tgt(tgt: Tensor, tgt_len: Tensor):
    """
    Flatten the target tensor based on the valid lengths of each target in the batch.
    
    Parameters:
    -----------
    tgt: Tensor of shape [B, max_tgt_len, num_attributes]
        The padded target tensor where each batch contains `max_tgt_len` targets.
    tgt_len: Tensor of shape [B]
        The tensor containing the number of valid targets for each batch.
        
    Returns:
    --------
    Tensor of shape [total_valid_targets, num_attributes]
        The flattened tensor containing only valid targets.
    """
    # Create a mask based on tgt_len (valid targets only)
    mask = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0) < tgt_len.unsqueeze(1)

    # Flatten the batch and use the mask to select only valid targets
    tgt_flat = tgt[mask]
    
    return tgt_flat
