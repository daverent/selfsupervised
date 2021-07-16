import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import torchvision

from utils.classificationmetrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

def imshow(img, transpose = True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def attribute_image_features(algorithm, input, classifier, labels, ind, **kwargs):
    classifier.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels[ind],
                                              **kwargs
                                             )
    return tensor_attributions

if __name__ == '__main__':

    validation_path = '../dataset/SelfSupervisedFULL/label'

    transform = transforms.Compose(
        [transforms.Resize((128,128)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    testset = datasets.ImageFolder(validation_path, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=12)

    classes = ('BAULETTO', 'CLUTCH', 'HOBO', 'MARSUPIO',
               'SACCA', 'SECCHIELLO', 'SHOPPING', 'TRACOLLA', 'ZAINO')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = models.resnet50(pretrained=False).to(device)
    net.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 9)).to(device)
    net.load_state_dict(torch.load('./weights.h5'))

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))

    net.cpu()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(8)))

    #ind = random.randrange(0, batch_size-1)
    ind = 7

    input = images[ind].unsqueeze(0).cuda()
    input.requires_grad = True

    net.eval()

    saliency = Saliency(net.cuda())
    grads = saliency.attribute(input, target=labels[ind].item()) #Va in stop qui
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    ig = IntegratedGradients(net)
    attr_ig, delta = attribute_image_features(ig, input, net, labels, ind, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    print('original Image')
    print('Predicted:', classes[predicted[ind]],
      'Probability: ', torch.max(F.softmax(outputs, 1)).item())
    original_image = np.transpose((images[ind].cpu().detach().numpy() /2)+0.5, (1, 2, 0))

    orig = viz.visualize_image_attr(None, original_image, method='original_image', title='Original Image')
    orig[0].savefig('../CaptumImages/Supervised_orig.jpg', format='jpg')

    overgrad = viz.visualize_image_attr(grads, original_image, method='blended_heat_map', sign='all', show_colorbar=True, title='Overlayed Gradient Magnitudes')
    overgrad[0].savefig('../CaptumImages/Supervised_saliency.png', format='png')

    overintgrad = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")
    overintgrad[0].savefig('../CaptumImages/Supervised_intgrad.png', format='png')

    want_confmat = True #Set for conf_mat output
    if want_confmat:
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        device=torch.device("cpu")
        net.cpu()
        with torch.no_grad():
            for i, (inputs, cm_classes) in enumerate(testloader):
                inputs = inputs.to(device)
                cm_classes = cm_classes.to(device)
                outputs = net(inputs)
                _, preds = torch.max(outputs, 1)

                # Append batch prediction results
                predlist=torch.cat([predlist,preds.cpu()])
                lbllist=torch.cat([lbllist,cm_classes.cpu()])

        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        print(conf_mat)
        figure = plot_confusion_matrix(conf_mat, classes)
        plt.savefig('../ConfusionMatrix/Supervised.pdf')
        print(classification_report(lbllist.numpy(), predlist.numpy(), zero_division=0))
        with open('../Reports/SupervisedClassificationReport.txt', mode='w') as savefile:
            savefile.write('Confusion Matrix:\n' + str(conf_mat))
            savefile.write('\n\n' + classification_report(lbllist.numpy(), predlist.numpy(), zero_division=0))

