import numpy as np
import torch
from torch.nn import functional as F
from matplotlib.lines import Line2D
from torch.autograd import Variable
from itertools import product


def dice_coeff(output, target, smooth=1, pred=False):
    if pred:
        pred = output
    else:
#         probs = F.softmax(output, dim=1)
        _, pred = torch.max(output, 1)

    if len(target.size()) == 5:
        target = target.squeeze(1)
    
    target = F.one_hot(target.long(), num_classes=output.size()[1])
    pred = F.one_hot(pred.long(), num_classes=output.size()[1])
    
    dim = tuple(range(1, len(pred.shape)-1))
    intersection = torch.sum(target * pred, dim=dim, dtype=torch.float)
    union = torch.sum(target, dim=dim, dtype=torch.float) + torch.sum(pred, dim=dim, dtype=torch.float)
    
    dice = torch.mean((2. * intersection + smooth)/(union + smooth), dtype=torch.float)
        
    return dice


def patchify(volume, patch_size, step):
    """

    :param volume:
    :param patch_size:
    :param step:
    :return:
    """
    assert len(volume.shape) == 4

    _, v_h, v_w, v_d = volume.shape

    s_h, s_w, s_d = step

    _, p_h, p_w, p_d = patch_size

    # Calculate the number of patch in each axis
    n_w = np.ceil(1.0*(v_w-p_w)/s_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/s_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/s_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_w = (n_w - 1) * s_w + p_w - v_w
    pad_h = (n_h - 1) * s_h + p_h - v_h
    pad_d = (n_d - 1) * s_d + p_d - v_d
    # print(volume.shape, (0, pad_h, 0, pad_w, 0, pad_d))
    volume = F.pad(volume, (0, pad_d, 0, pad_w, 0, pad_h), 'constant')
    # print(volume.shape)
    patches = torch.zeros((n_h, n_w, n_d,)+patch_size, dtype=volume.dtype)

    for i, j, k in product(range(n_h), range(n_w), range(n_d)):
        patches[i, j, k] = volume[:, (i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, (k * s_d):(k * s_d) + p_d]

    return patches

class Normalize(object):
    """Normalizes keypoints.
    """
    def __init__(self, max_val, min_val, new_max, new_min):
        self.max = max_val
        self.min = min_val
        self.new_max = new_max
        self.new_min = new_min
    
    def __call__(self, sample):

        image = (sample - self.min) * (self.new_max - self.new_min)/(self.max - self.min) + self.new_min
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return image



def unpatchify(patches, step, imsize, scale_factor):
    """

    :param patches:
    :param step:
    :param imsize:
    :param scale_factor:
    :return:
    """
    assert len(patches.shape) == 7

    c, r_h, r_w, r_d = imsize
    s_h, s_w, s_d = tuple(scale_factor*np.array(step))

    n_h, n_w, n_d, _, p_h, p_w, p_d = patches.shape

    v_w = (n_w - 1) * s_w + p_w
    v_h = (n_h - 1) * s_h + p_h
    v_d = (n_d - 1) * s_d + p_d

    volume = torch.zeros((c, v_h, v_w, v_d), dtype=patches.dtype)
    divisor = torch.zeros((c, v_h, v_w, v_d), dtype=patches.dtype)
#     print(volume.shape, imsize)

    for i, j, k in product(range(n_h), range(n_w), range(n_d)):
        patch = patches[i, j, k]
        volume[:, (i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, (k * s_d):(k * s_d) + p_d] += patch
        divisor[:, (i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, (k * s_d):(k * s_d) + p_d] += 1
    volume /= divisor
    return volume[:, 0:r_h, 0:r_w, 0:r_d]


def test(model, volume, patch_size=64, stride=60):
    model.eval()

    patch_size = (1, patch_size, patch_size, patch_size)
    stride = (stride, stride, stride)

    patches = patchify(volume, patch_size, stride)
    patch_shape = patches.shape
    patches = patches.view((-1,) + patch_size).cuda().type(torch.cuda.FloatTensor)

    output = torch.zeros((0, ) + patch_size[1:]).type(torch.FloatTensor)

    batch_size = 5 # user input
    num = int(np.ceil(1.0 * patches.shape[0] / batch_size))

    for i in range(num):
        model_output = model.forward(patches[batch_size*i:batch_size*i + batch_size])

        _, preds = torch.max(model_output, 1)
        preds = preds.cuda().type(torch.FloatTensor)

        output = torch.cat((output, preds), 0)

    new_shape = patch_shape
    output = unpatchify(output.view(new_shape), stride, volume.shape, 1)
    output = output.squeeze(0)
    
    return output