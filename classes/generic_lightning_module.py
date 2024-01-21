import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from classes.generic_network import GenericNetwork
from datetime import datetime
import matplotlib.pyplot as plt


class GenericLightningNetwork(pl.LightningModule):
    def __init__(self, parsed_layers, model_parameters, input_channels, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            model_parameters=model_parameters,
            input_channels=input_channels,
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
        plt.savefig(rf"./tb_logs/confusion_matrix_{current_datetime}.png")
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

    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer
