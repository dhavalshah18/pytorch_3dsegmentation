import torch
import torch.nn as nn
import numpy
import pathlib
from data_utils import MRAData
from trainer import Solver
from blocks import *
import os
import nibabel as nib

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_path = pathlib.Path("/home/dhaval/adam_data")
    
    size = [124, 124, 52]

    train_data = MRAData(dataset_path, patch_size=size, mode="train", suppress=True)
    val_data = MRAData(dataset_path, patch_size=size, mode="val", suppress=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=False, num_workers=2)
        
    model = UNet(in_channels=1, out_channels=2, final_activation = "sigmoid")
    model = model.cuda()

    optim_args_SGD = {"lr": 1e-1, "weight_decay": 0.0005, "momentum":0.9, "nesterov":True}

    optim_args_Adam = {"lr": 1e-2, "weight_decay": 0.0005}

    solver = Solver(optim_args=optim_args_SGD, optim=torch.optim.SGD)
    
    solver.train(model, train_loader, val_loader, log_nth=5, num_epochs=25)
    
    torch.save(model.state_dict(), "unet_1.pth")
    
if __name__ == "__main__":
    main()