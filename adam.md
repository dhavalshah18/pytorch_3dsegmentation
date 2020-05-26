#ADAM Challenge 2020
[Webapge for challenge](http://adam.isi.uu.nl/)

##3D UNet
As described in [this paper](https://arxiv.org/abs/1606.06650)

####Analysis / Encoder
(3 x 3 x 3) Conv3D  - (Batchnorm) - ReLU(ELU?) - (2 x 2 x 2) Max Pool
Depth? 2D is more deeper than 3D

####Synthesis / Decoder
(2 x 2 x 2) Upconvolution - (3 x 3 x 3) Conv3D - (Batchnorm) - ReLU (ELU?)

##To Do:
Get access to data and facebook server

Start design of network

Write down my timeline given the challenge schedule + thesis

Attention map 

Start with baseline 3D unet model, might try change the loss function later and add attention maps

##Concerns:
Data augmentations?
-- Pytorch 3d data augmentation library
-- Make sure to preserve global structure
-- Might be beneficial in our case because more global structure and not need to preserve fine details

Discuss about data, TOF will for sure be used to train but can I use the structural images? How?
-- Maybe don't need structural, add "vessel segmentation"

Same problem as before with 3D data, volume is too large, how to decide on which voxels to use to train
-- Random threshold for vessel segmentation before training

Registration? Both thesis and team for challenge
-- Registered team for challenge, can register thesis later

Structure of thesis, how to go about it and how it should end up; expectations etc
-- Usually around 30 pages, just more complicated bachelors

Possibility of getting past masters thesis from IBBM group