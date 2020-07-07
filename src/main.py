import torch
import torch.nn as nn
import numpy
import pathlib
from data_utils import MRAData
from solver import Solver
from network import *
import os
import time


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

#     torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_path = pathlib.Path("/home/dhaval/adam_data")
    
    size = [20, 20, 20]
    batch_size = 5
    optim = torch.optim.SGD
    lr = 2e-3
    wd = 0.005
    epochs = 50
    name = "r2attunet{0}.pth".format(size[0])


    train_data = MRAData(dataset_path, patch_size=size, mode="train")
    val_data = MRAData(dataset_path, patch_size=size, mode="val")
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=5)
#     val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=5)
        
    for i in train_data:
        i
        
#     model = nn.DataParallel(R2AttUNet(in_channels=1, out_channels=2))
#     model = model.cuda()
    
#     optim_args_SGD = {"lr": lr, "weight_decay": wd, "momentum":0.9, "nesterov":True}

#     solver = Solver(optim_args=optim_args_SGD, optim=optim)
    
#     start = time.time()
#     solver.train(model, train_loader, val_loader, log_nth=5, num_epochs=epochs)
#     end = time.time()
    
#     hours = (end - start)//(60*60)
#     mins = (((end - start)/(60*60)) - hours)*60
    
#     torch.save(model.state_dict(), name)
    
#     with open("saved_models.txt", "a") as savefile:
#         savefile.write("\n")
#         savefile.write("name={0}, patchsize={1}x{1}x{1}, batchsize={2}, optim={3}, lr={4}, wd={5}, epochs={6}, time={.0f}h{.0f}m"
#                        .format(name, size[0], batch_size, optim, lr, wd, epochs, hours, mins))


if __name__ == "__main__":
    main()