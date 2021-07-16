import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from captum.attr import DeepLift, IntegratedGradients, NoiseTunnel, Saliency
from captum.attr import visualization as viz


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

def captum_fun(classifier, dataset_test, dataloader_test, batch_size, classes, namestr):
    dataiter = iter(dataloader_test)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    outputs = classifier(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(batch_size)))

    #ind = random.randrange(0, batch_size-1)
    ind = 7

    input = images[ind].unsqueeze(0).cuda()
    input.requires_grad = True

    classifier.eval()

    saliency = Saliency(classifier.cuda())
    grads = saliency.attribute(input, target=labels[ind].item()) #Va in stop qui
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    ig = IntegratedGradients(classifier)
    attr_ig, delta = attribute_image_features(ig, input, classifier, labels, ind, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    print('original Image')
    print('Predicted:', classes[predicted[ind]],
      'Probability: ', torch.max(F.softmax(outputs, 1)).item())
    original_image = np.transpose((images[ind].cpu().detach().numpy() /2)+0.5, (1, 2, 0))

    orig = viz.visualize_image_attr(None, original_image, method='original_image', title='Original Image')
    orig[0].savefig('./CaptumImages/' + namestr + "_orig.jpg", format='jpg')

    overgrad = viz.visualize_image_attr(grads, original_image, method='blended_heat_map', sign='all', show_colorbar=True, title='Overlayed Gradient Magnitudes')
    overgrad[0].savefig('./CaptumImages/' + namestr + "_saliency.png", format='png')

    overintgrad = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")
    overintgrad[0].savefig('./CaptumImages/' + namestr + "_intgrad.png", format='png')
