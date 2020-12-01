# RAGNet
- The implementation of "Two-Stage Single Image Reflection Removal with Reflection-Aware Guidance".

# Prerequisites  
- The code has been test on a PC with following environment
  - Ubuntu 18.04
  - Python 3.7.5
  - PyTorch 1.2.0
  - cudatoolkit 10.0
  - NVIDIA RTX 2080Ti

# Datasets
### Training datasets
  - Synthetic: 7643 images from [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), image choices and the data synthesis protocol are same as [ERRNet](https://github.com/Vandermode/ERRNet).
  - Real: 90 real-world images from [Berkeley real dataset](https://github.com/ceciliavision/perceptual-reflection-removal).
### Testing datasets
  - Real20: 20 real testing images from [Berkeley real dataset](https://github.com/ceciliavision/perceptual-reflection-removal).
  - Real45: 45 real testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet).
  - SIR dataset: three sub-datasets (Solid, Postcard, Wild) from [SIR dataset](https://sir2data.github.io/).  
    
    We provide Real20 and Real45 in `./testsets` folder, the SIR dataset is not provided due to their policy, [download here](https://sir2data.github.io/) and put it under `./testsets` folder. Please organize the SIR dataset according to our code implementation.
  
# Test

- Download our [pre-trained model](https://drive.google.com/drive/folders/1qRyDQmh4mccejjK6A4OuWsy-L2ELyu0F?usp=sharing) and put the `pretrain.pth` into `./checkpoint` folder

### Test with the pre-trained model  
```shell
$ cd RAGNet
$ python test.py
```

# Train

- Download the [vgg19-pretrained model](https://drive.google.com/drive/folders/1qRyDQmh4mccejjK6A4OuWsy-L2ELyu0F?usp=sharing) and put it into `./checkpoint` folder.
- Organize the training dataset according to our code implementation, i.e.,
```shell
$ cd synthetic
$ mkdir transmission_layer
$ mkdir blended
```

### Start training
```shell
$ cd RAGNet
$ python train.py
```
