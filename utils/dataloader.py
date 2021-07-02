import torchvision
from pl_bolts.models.self_supervised.moco import Moco2TrainImagenetTransforms
from pl_bolts.models.self_supervised.swav.transforms import SwAVTrainDataTransform
from torch.utils.data import DataLoader
from pl_bolts.models.self_supervised import simclr

def create_dataloaders(configs):
    
    train_transforms = simclr.SimCLRTrainDataTransform(configs['img_input_size'])

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((configs['img_input_size'], configs['img_input_size'])),
        torchvision.transforms.ToTensor()
    ])
    
    dataset_train = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=train_transforms
    )

    dataset_train_classifier = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=transforms
    )

    dataset_test = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=transforms
    )

    dataloader_train = DataLoader(
                                dataset_train,
                                batch_size=configs['batch_size'],
                                shuffle=True,
                                num_workers=configs['num_workers']
                                )

    dataloader_train_classifier = DataLoader(
                                            dataset_train_classifier,
                                            batch_size=configs['batch_size'],
                                            shuffle=True,
                                            num_workers=configs['num_workers']
                                            )

    dataloader_test = DataLoader(
                                dataset_test,
                                batch_size=configs['batch_size'],
                                shuffle=False,
                                drop_last=False,
                                num_workers=configs['num_workers']
                                )


    classes = dataset_test.classes
    num_classes = len(classes)

    return dataloader_train, dataloader_train_classifier, dataloader_test, classes, num_classes

def create_moco_dataloaders(configs):
    
    train_transforms = Moco2TrainImagenetTransforms(configs['img_input_size'])

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((configs['img_input_size'], configs['img_input_size'])),
        torchvision.transforms.ToTensor()
    ])

    dataset_train = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=train_transforms
    )

    dataset_train_classifier = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=transforms
    )

    dataset_test = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=transforms
    )

    dataloader_train = DataLoader(
                                dataset_train,
                                batch_size=configs['batch_size'],
                                shuffle=True,
                                num_workers=configs['num_workers']
                                )

    dataloader_train_classifier = DataLoader(
                                            dataset_train_classifier,
                                            batch_size=configs['batch_size'],
                                            shuffle=True,
                                            num_workers=configs['num_workers']
                                            )

    dataloader_test = DataLoader(
                                dataset_test,
                                batch_size=configs['batch_size'],
                                shuffle=False,
                                drop_last=False,
                                num_workers=configs['num_workers']
                                )


    classes = dataset_test.classes
    num_classes = len(classes)

    return dataloader_train, dataloader_train_classifier, dataloader_test, classes, num_classes

def create_swav_dataloaders(configs):
    
    train_transforms = SwAVTrainDataTransform()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((configs['img_input_size'], configs['img_input_size'])),
        torchvision.transforms.ToTensor()
    ])
    
    dataset_train = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=train_transforms
    )

    dataset_train_classifier = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=transforms
    )

    dataset_test = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=transforms
    )

    dataloader_train = DataLoader(
                                dataset_train,
                                batch_size=configs['batch_size'],
                                shuffle=True,
                                num_workers=configs['num_workers']
                                )

    dataloader_train_classifier = DataLoader(
                                            dataset_train_classifier,
                                            batch_size=configs['batch_size'],
                                            shuffle=True,
                                            num_workers=configs['num_workers']
                                            )

    dataloader_test = DataLoader(
                                dataset_test,
                                batch_size=configs['batch_size'],
                                shuffle=True,
                                drop_last=False,
                                num_workers=configs['num_workers']
                                )


    classes = dataset_test.classes
    num_classes = len(classes)

    return dataloader_train, dataloader_train_classifier, dataloader_test, classes, num_classes

def create_dataset_test(configs):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((configs['img_input_size'], configs['img_input_size'])),
        torchvision.transforms.ToTensor()
    ])

    dataset_test = torchvision.datasets.ImageFolder(
        configs['path_to_test'],
        transform=transforms
    )

    dataloader_test = DataLoader(
                                dataset_test,
                                batch_size=configs['batch_size'],
                                shuffle=True,
                                drop_last=False,
                                num_workers=configs['num_workers']
                                )

    return dataloader_test, dataset_test