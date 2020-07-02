import torch
import torch.nn as nn
import numpy
import pathlib
from data_utils import MRAData
from trainer import Solver
from blocks import *
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_path = pathlib.Path("/home/dhaval/adam_data")
    
    size = [60, 60, 60]
    batch_size = 5
    optim = torch.optim.SGD
    lr = 5e-2
    wd = 0.000
    epochs = 50
    name = "unet{0}_{1}.pth".format(size[0], lr)


    train_data = MRAData(dataset_path, patch_size=size, mode="train")
    val_data = MRAData(dataset_path, patch_size=size, mode="val")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
        
    model = UNet(in_channels=1, out_channels=2, final_activation ="softmax")
    model = model.cuda()
    
    optim_args_SGD = {"lr": lr, "weight_decay": wd, "momentum":0.9, "nesterov":True}

    solver = Solver(optim_args=optim_args_SGD, optim=optim)
    
    solver.train(model, train_loader, val_loader, log_nth=5, num_epochs=epochs)
    
    torch.save(model.state_dict(), name)
    
    with open("saved_models.txt", "a") as savefile:
        savefile.write("\n")
        savefile.write("name={0}, patchsize={1}x{1}x{1}, batchsize={2}, optim={3}, lr={4}, wd={5}, epochs={6}".format(name, size[0], batch_size, optim, lr, wd, epochs))

if __name__ == "__main__":
    main()