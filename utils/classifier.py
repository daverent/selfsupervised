import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAClassifier(pl.LightningModule):
    def __init__(self, 
                backbone,
                num_classes,
                input_dim: int = 2048
                ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():  # reset requires_grad
             p.requires_grad = False
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)[0] 
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class Classifier(pl.LightningModule):
    def __init__(self, 
                backbone,
                num_classes,
                input_dim: int = 2048
                ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():  # reset requires_grad
             p.requires_grad = False
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    # def __init__(self, 
    #             backbone,
    #             num_classes,
    #             input_dim: int = 2048
    #             ):
    #     super().__init__()
    #     self.backbone = backbone
    #     for p in self.backbone.parameters():  # reset requires_grad
    #          p.requires_grad = False
    #     self.fc = nn.Linear(in_features=input_dim, out_features=num_classes)
    
    # def forward(self, x):
    #     with torch.no_grad():
    #         x = self.backbone(x)
    #         x = F.relu(x)
    #     x = torch.flatten(x, start_dim=1)
    #     x = self.fc(x)
    #     x = F.log_softmax(x, dim=1)
    #     return x
    
    # def custom_histogram_weights(self):
    #     for name, params in self.named_parameters():
    #         self.logger.experiment.add_histogram(
    #             name, params, self.current_epoch)

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)
    #     self.log('train_loss', loss)
    #     return loss

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-3)
