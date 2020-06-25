import pathlib
import numpy as np
import torch
import torch.utils.data as data
# from torchvision import transforms
import nibabel as nib
from misc import Normalize

class MRAData(data.Dataset):
    """ 
    Class defined to handle MRA data
    derived from pytorch's Dataset class.
    """
    
    def __init__(self, root_path, transform="normalize", 
                 patch_size=[132, 132, 116], mode="train"):
        # Note: Should already be split into train, val and test folders
        # TODO change to accept any kind of folder
        self.root_dir = pathlib.Path(root_path).joinpath(mode)
        
        # Array of paths of all case folders in root path
        self.cases_dirs = [cases for cases in sorted(self.root_dir.glob("100*/"))]
        
        self.transform = transform
        
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size, patch_size]
        
        self.patch_size = patch_size
        self.mode = mode
        
        
    def __len__(self):
        return len(self.cases_dirs)
    
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
        load_path = self.cases_dirs[index]
        
        # TODO: Orig or Pre??
        raw_vol_path = load_path.joinpath("pre/TOF.nii.gz")
        seg_vol_path = load_path.joinpath("aneurysms.nii.gz")
        
        # Load proxy so image not loaded into memory
        raw_proxy = nib.load(str(raw_vol_path))
        seg_proxy = nib.load(str(seg_vol_path))
           
        # Get dataobj of proxy
        raw_data = np.asarray(raw_proxy.dataobj).astype(np.int32)
        seg_data = np.asarray(seg_proxy.dataobj).astype(np.int32)

        raw_image = torch.from_numpy(raw_data)
        seg_image = torch.from_numpy(seg_data)
        
        # If training only return single patch
        if self.mode == "train":
            location_path = load_path.joinpath("location.txt")

            # Get center of aneurysm from location.txt file of case
            aneurysm_location_coords = self.get_aneurysm_coords(location_path)
                        
            # Get patch from centre of location of aneurysm
            raw_image, seg_image = self.get_patch(raw_image, seg_image, aneurysm_location_coords)
            
            if self.transform == "normalize":
                normalize = Normalize(torch.max(raw_image), torch.min(raw_image), 255., 0.)
                raw_image = normalize(raw_image)
        
        if self.mode == "val":
            # Resize val
            location_path = load_path.joinpath("location.txt")

            # Get center of aneurysm from location.txt file of case
            aneurysm_location_coords = self.get_aneurysm_coords(location_path)
                        
            # Get patch from centre of location of aneurysm
            raw_image, seg_image = self.get_patch(raw_image, seg_image, aneurysm_location_coords)
            
            if self.transform == "normalize":
                normalize = Normalize(torch.max(raw_image), torch.min(raw_image), 255., 0.)
                raw_image = normalize(raw_image)

        return raw_image, seg_image
    
    def get_aneurysm_coords(self, location_path):
        location_file = open(str(location_path), "r").readlines()
        location_coords = []
        
        for i in range(len(location_file)):
            aneurysm = list(map(float, location_file[i].split(", ")))
            location_coords.append(aneurysm)
        
        return location_coords
    
    def get_patch(self, raw_image, seg_image, location_coords):
        if len(location_coords) == 0:
            # If no aneurysm, return random patch
            raw_patch = raw_image. \
                unfold(2, self.patch_size[2], self.patch_size[2]). \
                unfold(1, self.patch_size[1], self.patch_size[1]). \
                unfold(0, self.patch_size[0], self.patch_size[0])
            raw_patch = raw_patch.contiguous(). \
                view(-1, 1, self.patch_size[0], self.patch_size[1], self.patch_size[2])

            seg_patch = seg_image. \
                unfold(2, self.patch_size[2], self.patch_size[2]). \
                unfold(1, self.patch_size[1], self.patch_size[1]). \
                unfold(0, self.patch_size[0], self.patch_size[0])
            seg_patch = seg_patch.contiguous(). \
                view(-1, 1, self.patch_size[0], self.patch_size[1], self.patch_size[2])
            
            patch_num = np.random.randint(seg_patch.shape[0])
            
            raw_patch = raw_patch[patch_num]
            seg_patch = seg_patch[patch_num]
            
        elif len(location_coords) == 1:
            # Check if out of bounds
            slices_min = np.clip(np.array(location_coords[0])[:3] 
                                 - np.array(self.patch_size)//2, 0, None)
            slices_min = slices_min.astype(int)
            slices_max = np.clip(np.array(location_coords[0])[:3] 
                                 + np.array(self.patch_size)//2, None, raw_image.size())
            slices_max = slices_max.astype(int)
            
            # Make sure slice is properly sized
            for i in range(3):
                rem = self.patch_size[i] - (slices_max[i] - slices_min[i])
                if rem and slices_max[i] < raw_image.size()[i]:
                    slices_max[i] += rem
                elif rem and slices_min[i] > 0:
                    slices_min[i] -= rem
                    
                assert (slices_max[i] - slices_min[i] == self.patch_size[i]), "Mis-sized slice"
                    
            
            raw_patch = raw_image[slices_min[0]:slices_max[0],
                                  slices_min[1]:slices_max[1],
                                  slices_min[2]:slices_max[2]]
            raw_patch = raw_patch.unsqueeze(0)

            seg_patch = seg_image[slices_min[0]:slices_max[0],
                                  slices_min[1]:slices_max[1],
                                  slices_min[2]:slices_max[2]]
            seg_patch = seg_patch.unsqueeze(0)
            
            
        elif len(location_coords) > 1:
            coords = np.array(location_coords).astype(int)
            mins = np.amin(coords, 0)[:3]
            maxs = np.amax(coords, 0)[:3]

            remainders = self.patch_size - (maxs - mins)
            
            slices_min = np.clip(mins - remainders//2, 0, None)
            slices_max = np.clip(maxs + (remainders - remainders//2), None, raw_image.size())
            
            for i in range(3):
                rem = self.patch_size[i] - (slices_max[i] - slices_min[i])
                if rem and slices_max[i] < raw_image.size()[i]:
                    slices_max[i] += rem
                elif rem and slices_min[i] > 0:
                    slices_min[i] -= rem
                
                assert (slices_max[i] - slices_min[i] == self.patch_size[i]), "Mis-sized slice"

            
            raw_patch = raw_image[slices_min[0]:slices_max[0], 
                                  slices_min[1]:slices_max[1],
                                  slices_min[2]:slices_max[2]]
            raw_patch = raw_patch.unsqueeze(0)

            seg_patch = seg_image[slices_min[0]:slices_max[0], 
                                  slices_min[1]:slices_max[1],
                                  slices_min[2]:slices_max[2]]
            seg_patch = seg_patch.unsqueeze(0)
            
        return raw_patch, seg_patch
                                  
