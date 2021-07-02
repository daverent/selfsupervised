from BYOL.byol import use_BYOL
from MoCo.moco import use_MoCo
from SimCLR.simclr import use_SimCLR
from SimSiam.simsiam import use_SimSiam
import pytorch_lightning as pl

if __name__ == '__main__':
    # Configs
    num_workers = 12
    batch_size = 8
    max_epochs = 2
    img_input_size = 128
    gpus = 1
    seed = 1

    pl.seed_everything(seed)

    path_to_train = './dataset/SelfSupervised/nolabel'
    path_to_test = './dataset/SelfSupervised/label'

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
        choice = input("\nPlease, choose a model from the list (quit: q):\n\t[BYOL: b,\n\t MoCo: m,\n\t SimCLR: sc,\n\t SimSiam: ss,\n\t All: a]\n")

        if choice=='b':
            print("\n\nBYOL:\n")
            use_BYOL(configs)
        if choice=='m':
            print("\n\nMoCo:\n")
            use_MoCo(configs)
        if choice=='sc':
            print("\n\nSimCLR:\n")
            use_SimCLR(configs)
        if choice=='ss':
            print("\n\nSimSiam:\n")
            use_SimSiam(configs)
        if choice=='a':
            print("\n\nGo play pokèmon, it's gonna take some time...")
            use_BYOL(configs)
            use_MoCo(configs)
            use_SimCLR(configs)
            use_SimSiam(configs)
        if choice=='q':
            check=False
        


