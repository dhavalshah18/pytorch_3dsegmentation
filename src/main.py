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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

    device = torch.cuda.set_device("cuda:1")

    dataset_path = pathlib.Path("/home/dhaval/adam_data")

    train_data = MRAData(dataset_path, patch_size=[100, 100, 60], mode="train")
    val_data = MRAData(dataset_path, patch_size=[100, 100, 60], mode="val")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=3, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=False, num_workers=2)
    
    model = UNet(in_channels=1, out_channels=3)
    model.to(device).cuda()

    optim_args_SGD = {"lr": 1e-3, "weight_decay": 0.0005, "momentum": 0.9, "nesterov": True}

    solver = Solver(optim_args=optim_args_SGD, optim=torch.optim.SGD)
    solver.train(model, train_loader, val_loader, log_nth=2, num_epochs=50)
    
    torch.save(model.state_dict(), "unet.pth")
    
if __name__ == "__main__":
    main()