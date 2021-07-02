import os
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000 # Allow decompression bomb

from utils.dataloader import create_dataloaders, create_dataset_test
from utils.classifier import SAClassifier
from utils.classificationmetrics import plot_confusion_matrix
from utils.captum_file import captum_fun


def use_BYOL(configs):

    max_epochs = configs['max_epochs']
    batch_size = configs['batch_size']
    gpus = configs['gpus']

    dataloader_train, dataloader_train_classifier, dataloader_test, classes, num_classes = create_dataloaders(configs)

    model_path = "./BYOL/Models/BYOL_"+str(configs['path_to_train'].split('/')[-2])+"_"+str(configs['max_epochs'])+"ep.pt"
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("\nModel loaded from previous training.\n")
    else:    
        resnet = torchvision_ssl_encoder('resnet50', pretrained=True)
        model = BYOL(num_classes, base_encoder=resnet, batch_size=batch_size)
        logger = TensorBoardLogger("tb_logs", name="BYOL")
        trainer = pl.Trainer(max_epochs=max_epochs,
                        gpus=gpus,
                        progress_bar_refresh_rate=5,
                        logger=logger)
        trainer.fit(model, dataloader_train)
        torch.save(model, model_path)

    classifier_path = "./BYOL/Models/BYOLClassifier_"+str(configs['path_to_train'].split('/')[-2])+"_"+str(configs['max_epochs'])+"ep.pt"
    model.eval()
    if os.path.exists(classifier_path):
        classifier = torch.load(classifier_path)
    else:
        classifier = SAClassifier(model.online_network, num_classes=num_classes)
        classifier_logger = TensorBoardLogger("tb_logs", name="BYOLClassifier")
        classifier_trainer = pl.Trainer(max_epochs=max_epochs,
                                        gpus=gpus,
                                        progress_bar_refresh_rate=5,
                                        logger=classifier_logger)
        classifier_trainer.fit(classifier, dataloader_train_classifier)
        torch.save(classifier, classifier_path)

    #confusion Matrix
    want_confmat = False #Set for conf_mat output
    if want_confmat:
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        device=torch.device("cpu")

        with torch.no_grad():
            for i, (inputs, classes) in enumerate(dataloader_test):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = classifier(inputs)
                _, preds = torch.max(outputs, 1)

                # Append batch prediction results
                predlist=torch.cat([predlist,preds.cpu()])
                lbllist=torch.cat([lbllist,classes.cpu()])

        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        print(conf_mat)
        figure = plot_confusion_matrix(conf_mat)
        plt.savefig('./ConfusionMatrix/BYOL_'+str(configs['max_epochs'])+'ep.pdf')
        print(classification_report(lbllist.numpy(), predlist.numpy(), zero_division=0))

    #Captum call
    _, dataset_test = create_dataset_test(configs=configs)
    captum_fun(classifier, dataset_test=dataset_test, dataloader_test=dataloader_test, batch_size=configs['batch_size'], classes=classes)

    return model, classifier

