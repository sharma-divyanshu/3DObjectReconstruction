# Single and Multi-view 3D Object Reconstruction

Inspired by methods that employ shape priors to achieve
robust 3D reconstructions, in this project we implement a
recurrent neural network architecture called the 3D
Recurrent Reconstruction Neural Network. The network
learns a mapping from images of objects to their underlying
3D shapes from a large collection of synthetic 3D models.
The network takes in one or more images of an object
instance from arbitrary viewpoints and outputs a
reconstruction of the object in the form of a 3D occupancy
voxel grid. The network does not require any image
annotations or object class labels for training or testing.
The reconstruction framework enables 3D reconstruction
of objects in situations when traditional SFM/SLAM
methods fail because of lack of texture and/or wide
baseline.

Reference:
Christopher B. Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, Silvio Savarese. 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction https://arxiv.org/abs/1604.00449v1
