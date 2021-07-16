import os

import matplotlib.pyplot as plt
import PIL
import pytorch_lightning as pl
import torch
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 933120000 # Allow decompression bomb

from utils.classificationmetrics import plot_confusion_matrix
from utils.classifier import SAClassifier
from utils.dataloader import create_dataloaders


def use_SimCLR(configs):

    dataloader_train, dataloader_train_classifier, dataloader_test, classes, num_classes = create_dataloaders(configs)

    model_path = "./SimCLR/Models/SimCLR_"+str(configs['path_to_train'].split('/')[-2])+"_"+str(configs['max_epochs'])+"ep.pt" #Standard torch.save() doesn't work, saving state_dict
    if os.path.exists(model_path):
        model = SimCLR(gpus=configs['gpus'], num_samples=num_classes, batch_size=configs['batch_size'], dataset=configs['path_to_train'].split('/')[-2], feat_dim=2048)
        model.load_state_dict(state_dict=torch.load(model_path))
        print('Model loaded from previous training')
    else:    
        model = SimCLR(gpus=configs['gpus'], num_samples=num_classes, batch_size=configs['batch_size'], dataset=configs['path_to_train'].split('/')[-2], feat_dim=2048)
        logger = TensorBoardLogger("tb_logs", name="SimCLR")
        trainer = pl.Trainer(max_epochs=configs['max_epochs'],
                            gpus=configs['gpus'],
                            progress_bar_refresh_rate=5,
                            logger=logger)
        trainer.fit(model, dataloader_train)
        torch.save(model.state_dict(), model_path)

    classifier_path = "./SimCLR/Models/SimCLRClassifier_"+str(configs['path_to_train'].split('/')[-2])+"_"+str(configs['max_epochs'])+"ep.pt"
    if os.path.exists(classifier_path):
        classifier = SAClassifier(backbone=model.encoder, num_classes=num_classes)
        classifier.load_state_dict(state_dict=torch.load(classifier_path))
        print('Classifier loaded from previous training')
    else:
        classifier = SAClassifier(backbone=model.encoder, num_classes=num_classes)
        classifier_logger = TensorBoardLogger("tb_logs", name="SimCLRClassifier")
        classifier_trainer = pl.Trainer(max_epochs=int(configs['max_epochs']),
                                        gpus=configs['gpus'],
                                        progress_bar_refresh_rate=5,
                                        logger=classifier_logger)
        classifier_trainer.fit(classifier, dataloader_train_classifier)
        torch.save(classifier.state_dict(), classifier_path)

    #confusion Matrix
    want_confmat = True #Set for conf_mat output
    if want_confmat:
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        device=torch.device("cpu")

        with torch.no_grad():
            for i, (inputs, cm_classes) in tqdm(enumerate(dataloader_test)):
                inputs = inputs.to(device)
                cm_classes = cm_classes.to(device)
                outputs = classifier(inputs)
                _, preds = torch.max(outputs, 1)

                # Append batch prediction results
                predlist=torch.cat([predlist,preds.cpu()])
                lbllist=torch.cat([lbllist,cm_classes.cpu()])

        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        print(conf_mat)
        figure = plot_confusion_matrix(conf_mat, classes)
        plt.savefig('./ConfusionMatrix/SimCLR_'+str(configs['max_epochs'])+'ep.pdf')
        print(classification_report(lbllist.numpy(), predlist.numpy(), zero_division=0))
        with open('./Reports/SimCLR_'+ str(configs['max_epochs']) +'ep_ClassificationReport.txt', mode='w') as savefile:
            savefile.write('Confusion Matrix:\n' + str(conf_mat))
            savefile.write('\n\n' + classification_report(lbllist.numpy(), predlist.numpy(), zero_division=0))
    
    return model, classifier
