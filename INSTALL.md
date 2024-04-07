# Installation

### Dependencies Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 18.04 environment (Python3.8, CUDA11.6, cuDNN8.5).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/va1shn9v/PromptIR.git
cd PromptIR
```

2. Create conda environment
The Conda environment used can be recreated using the env.yml file
```
conda env create -f env.yml
```


### Dataset Download and Preperation

All the 5 datasets used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise,Derain or Dehaze.
After placing the training data the directory structure would be as follows:
```
└───Train
    ├───Dehaze
    │   ├───original
    │   └───synthetic
    ├───Denoise
    └───Derain
        ├───gt
        └───rainy
```

The testing data should be placed in the ```test``` directory wherein each task has a seperate directory. The test directory after setup:

```
├───dehaze
│   ├───input
│   └───target
├───denoise
│   ├───bsd68
│   └───urban100
└───derain
    └───Rain100L
        ├───input
        └───target
```
