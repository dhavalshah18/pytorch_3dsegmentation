import pathlib
import os
import numpy as np
import torch
import torch.nn as nn
import misc as ms
import metrics
from data_utils import MRAData
from network import *

def test_single(model, volume, patch_size=60, stride=56, batch_size=2):
    model.eval()

    patch_size = (1, patch_size, patch_size, patch_size)
    stride = (stride, stride, stride)

    patches = ms.patchify(volume, patch_size, stride)
    patch_shape = patches.shape
    patches = patches.view((-1,) + patch_size).cuda().type(torch.cuda.FloatTensor)

    output = torch.zeros((0, ) + patch_size[1:]).type(torch.FloatTensor)

    batch_size = batch_size # user input
    num = int(np.ceil(1.0 * patches.shape[0] / batch_size))

    for i in range(num):
#         print("{0} / {1}".format(i, num))
        curr_patch = patches[batch_size*i:batch_size*i + batch_size]
        
        model_output = model(curr_patch)

        _, preds = torch.max(model_output, 1)
        preds = preds.cuda().type(torch.FloatTensor)

        output = torch.cat((output, preds), 0)

    new_shape = patch_shape
    output = ms.unpatchify(output.view(new_shape), stride, volume.shape, 1)
    output = output.squeeze(0)
    
    return output

def test(model, test_loader):
    # Note: make sure batch size for test loader is 1
    model.eval()
    
    test_dice_scores = []
    test_hausdorff = []
    
    print("START EVAL")
    for i, (inputs, target, pixdim) in enumerate(test_loader):
        output = test_single(model, inputs)
        output = output.detach()
        
#         dice = ms.dice_coeff(output, target).detach().cpu().numpy()
        
        ones, zeros = torch.ones_like(output), torch.zeros_like(output)
        output = torch.where(output==1, ones, zeros)
        output = output.numpy().astype(bool)
        target = target.squeeze(0).detach().numpy().astype(bool)
        
#         print(list(pixdim.detach().numpy()))
        print("Pred: ", np.count_nonzero(output))
        print("GT: ", np.count_nonzero(target))
        
#         surface_distances = metrics.compute_surface_distances(target, output, pixdim)
#         hausdorff = metrics.compute_robust_hausdorff(surface_distances, 95.)
        
#         print("Dice/Hausdorff: {0:.2f}/{1:.2f}".format(dice, hausdorff))
        
#         test_dice_scores.append(dice)
#         test_hausdorff.append(hausdorff)
        
#     print("Average Dice: ", sum(test_dice_scores)/len(test_dice_scores))
#     print("Average Hausdorff 95%: ", sum(test_hausdorff)/len(test_hausdorff))
        

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5"
    model_path = pathlib.Path("/home/dhaval/pytorch_3dunet/r2attunet_newdata28.pth")
    dataset_path = pathlib.Path("/home/dhaval/adam_data")

    model = nn.DataParallel(R2AttUNet(in_channels=1, out_channels=2))
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
#     model.eval()
    
    test_data = MRAData(dataset_path, mode="test", transform="")[5:]
    # No batches for test loader
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=False, num_workers=5)
    
    test(model, test_loader)
    
    