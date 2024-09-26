# Deep Lidar-Guided Image Deblurring
This repository contains the official implementation for "Deep Lidar-Guided Image Deblurring"

## Citation
If you find our work useful in your research, please consider citing: 

```
@inproceedings{ziyao2024deep,
 author = {Yi, Ziyao and Valsesia, Diego and Bianchi, Tiziano and Magli, Enrico},
 booktitle = {in peer review},
 title = {},
 year = {2024}
}
```

## Introduction
In this paper, we study if the depth information provided by mobile Lidar sensors is useful for the task of image deblurring and how to integrate it with a general approach that transforms any state-of-the-art neural deblurring model into a depth-aware one. To achieve this, we developed a universal adapter structure that efficiently preprocesses the depth information to modulate image features with depth features. Additionally, we applied a continual learning strategy to pretrained encoder-decoder models, enabling them to incorporate depth information as an additional input with minimal extra data requirements. We demonstrate that utilizing true depth information can significantly boost the effectiveness of deblurring algorithms, as validated on a dataset with real-world depth data captured by a smartphone Lidar.


## Requirements
The code has been tested with the following dependencies:
- python 3.10
- pytorch 2.0.1
- natten (for DeblurDiNATL)


## Dataset
The code uses the ARKitScenes datasets, available for download from [https://github.com/apple/ARKitScenes](https://github.com/apple/ARKitScenes)

## How to train
1. config your train hyperparameter in config/*.yaml 
2. run 
```python main.py -task train -model_type original -model_task Deblur/DepthDeblur/KernelDeblur/KernelDepthDeblur -device cuda/cpu
```

## How to test
1. config your train hyperparameter in config/*.yaml 
2. run 
```python main.py -task test -model_type original -model_task Deblur/DepthDeblur/KernelDeblur/KernelDepthDeblur -device cuda/cpu
```


## Pre-trained Models

You can download pretrained models here: [https://www.dropbox.com/scl/fo/1vdmlh64yhs3dr1ilkowf/AJXGJdKxE7VHG9D5ss_XUiM?rlkey=27874gc4gkj6qgoxisqzggpae&st=j6l55k7z&dl=0](https://www.dropbox.com/scl/fo/1vdmlh64yhs3dr1ilkowf/AJXGJdKxE7VHG9D5ss_XUiM?rlkey=27874gc4gkj6qgoxisqzggpae&st=j6l55k7z&dl=0)


## Acknowledgement
This study was carried out within the “AI-powered LIDAR fusion for next-generation smartphone cameras (LICAM)” project – funded by European Union – Next Generation EU within the PRIN 2022 program (D.D. 104 - 02/02/2022 Ministero dell’Università e della Ricerca). This manuscript reflects only the authors' views and opinions and the Ministry cannot be considered responsible for them.


## License 
Our code is released under MIT License (see LICENSE file for details).



