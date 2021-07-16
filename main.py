import pytorch_lightning as pl

from BYOL.byol import use_BYOL
from MoCo.moco import use_MoCo
from SimCLR.simclr import use_SimCLR
from SimSiam.simsiam import use_SimSiam
from utils.captum_file import captum_fun
from utils.dataloader import create_dataset_test

if __name__ == '__main__':
    # Configs
    num_workers = 12
    batch_size = 8
    max_epochs = 180
    img_input_size = 128
    gpus = 1
    seed = 1

    pl.seed_everything(seed)

    path_to_train = './dataset/SelfSupervisedFULL/nolabel'
    path_to_test = './dataset/SelfSupervisedFULL/label'

    #TODO: Input function to change configs at runtime

    configs = {
        'num_workers': num_workers,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'img_input_size': img_input_size,
        'gpus': gpus,
        'path_to_train': path_to_train,
        'path_to_test': path_to_test
    }

    check = True
    print("This software is made to train self-supervised models and study their outputs.\n")
    print(configs)
    while(check):
        choice = 'q'
        choice = input("\nPlease, choose a model from the list (quit: q):\n\t[BYOL: b,\n\t MoCo: m,\n\t SimCLR: sc,\n\t SimSiam: ss,\n\t Train all: a]\n")

        if choice=='b':
            print("\n\nBYOL:\n")
            _, classifier = use_BYOL(configs)

            choice2 = input("\nShow Captum network interpretation?[y,_]\n")
            if choice2 =='y':
                #Captum call
                dataloader_test, dataset_test = create_dataset_test(configs=configs)
                captum_fun(classifier, 
                    dataset_test=dataset_test, 
                    dataloader_test=dataloader_test, 
                    batch_size=configs['batch_size'], 
                    classes=dataset_test.classes, 
                    namestr= "BYOL_"+str(max_epochs)+"ep")
        
        if choice=='m':
            print("\n\nMoCo:\n")
            _, classifier = use_MoCo(configs)
            choice2 = input("\nShow Captum network interpretation?[y,_]\n")
            if choice2 =='y':
                #Captum call
                dataloader_test, dataset_test = create_dataset_test(configs=configs)
                captum_fun(classifier, 
                    dataset_test=dataset_test, 
                    dataloader_test=dataloader_test, 
                    batch_size=configs['batch_size'], 
                    classes=dataset_test.classes, 
                    namestr= "MoCo_"+str(max_epochs)+"ep")
        
        if choice=='sc':
            print("\n\nSimCLR:\n")
            _, classifier = use_SimCLR(configs)

            choice2 = input("\nShow Captum network interpretation?[y,_]\n")
            if choice2 =='y':
                #Captum call
                dataloader_test, dataset_test = create_dataset_test(configs=configs)
                captum_fun(classifier, 
                    dataset_test=dataset_test, 
                    dataloader_test=dataloader_test, 
                    batch_size=configs['batch_size'], 
                    classes=dataset_test.classes, 
                    namestr= "SimCLR_"+str(max_epochs)+"ep")
        
        if choice=='ss':
            print("\n\nSimSiam:\n")
            _, classifier = use_SimSiam(configs)

            choice2 = input("\nShow Captum network interpretation?[y,_]\n")
            if choice2 =='y':
                #Captum call
                dataloader_test, dataset_test = create_dataset_test(configs=configs)
                captum_fun(classifier, 
                    dataset_test=dataset_test, 
                    dataloader_test=dataloader_test, 
                    batch_size=configs['batch_size'], 
                    classes=dataset_test.classes, 
                    namestr= "SimSiam_"+str(max_epochs)+"ep")
        
        if choice=='a':
            print("\n\nGo play pokèmon, it's gonna take some time...\n")
            use_BYOL(configs)
            #use_MoCo(configs)
            use_SimCLR(configs)
            use_SimSiam(configs)
        if choice=='q':
            check=False
            break
