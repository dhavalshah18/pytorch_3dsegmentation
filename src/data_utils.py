import pathlib
import numpy as np
import torch
import torch.utils.data as data
# from torchvision import transforms
import nibabel as nib

from dvn import patchify_unpatchify as pu
from dvn import transforms

class MRAData(data.Dataset):
    """ 
    Class defined to handle MRA data
    derived from pytorch's Dataset class.
    """
    
    def __init__(self, root_path, transform="normalize", 
                 patch_size=64, stride=60, mode="train"):
        # Note: Should already be split into train, val and test folders
        # TODO change to accept any kind of folder
        self.root_dir = pathlib.Path(root_path).joinpath(mode)
        
        # Array of paths of all case folders in root path
        self.cases_dirs = [cases for cases in sorted(self.root_dir.glob("*/"))]
        
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        
        
    def __len__(self):
        return len(self.orig_dirs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            return [self[i] for i in range(index)]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index > len(self):
                raise IndexError("The index (%d) is out of range." % index)
            
            # get the data from direct index
            return self.get_item_from_index(index)

        else:
            raise TypeError("Invalid argument type.")
    
    def get_item_from_index(self, index):
        load_path = self.cases_dir[index]
        
        # TODO: Orig or Pre??
        raw_vol_path = load_path.joinpath("/orig/TOF.nii.gz")
        seg_vol_path = load_path.joinpath("aneurysm.nii.gz")
        
        # Load proxy so image not loaded into memory
        raw_proxy = nib.load(str(raw_vol_path))
        seg_proxy = nib.load(str(seg_vol_name))

        # Get dataobj of proxy
        raw_data = np.asarray(raw_proxy.dataobj).astype(np.int32)
        seg_data = np.asarray(seg_proxy.dataobj).astype(np.int32)
#         print("Num of seg pixels: ", np.argwhere(seg_data == 1).size)

        raw_image = torch.from_numpy(raw_data)
        
        seg_image = torch.from_numpy(seg_data)
        
        # If training only return single patch
        if self.mode == "train":
            raw_patch = raw_image. \
                unfold(2, self.patch_size, self.patch_size). \
                unfold(1, self.patch_size, self.patch_size). \
                unfold(0, self.patch_size, self.patch_size)
            raw_patch = raw_patch.contiguous(). \
                view(-1, 1, self.patch_size, self.patch_size, self.patch_size)

            seg_patch = seg_image. \
                unfold(2, self.patch_size, self.patch_size). \
                unfold(1, self.patch_size, self.patch_size). \
                unfold(0, self.patch_size, self.patch_size)
            seg_patch = seg_patch.contiguous(). \
                view(-1, self.patch_size, self.patch_size, self.patch_size)
            
            # TODO rework
            # Random number to select patch number
            patch_num = np.random.randint(seg_patch.shape[0])
            
            raw_image = raw_patch[patch_num]
            seg_image = seg_patch[patch_num]
            
        
        if self.transform == "normalize":
            normalize = transforms.Normalize(torch.max(raw_image), torch.min(raw_image), 0., 255.)
            raw_image = normalize(raw_image)
        
        return raw_image, seg_image
    
    