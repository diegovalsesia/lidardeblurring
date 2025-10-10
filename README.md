![](logo.jpeg)

# Deep Lidar-Guided Image Deblurring
This repository contains the official implementation for "Deep Lidar-Guided Image Deblurring"


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
```
python main.py -task train -model_type original -model_task Deblur/DepthDeblur/KernelDeblur/KernelDepthDeblur -device cuda/cpu
```

## How to test
1. config your train hyperparameter in config/*.yaml 
2. run 
```
python main.py -task test -model_type original -model_task Deblur/DepthDeblur/KernelDeblur/KernelDepthDeblur -device cuda/cpu
```


## Pre-trained Models

You can download pretrained models here: [https://www.dropbox.com/scl/fo/wxwa943qyeisgmz0sp2wv/AKEz-DIw8g8kI9bKRLwN9j8?rlkey=a6nghj80ogl7gzjb7re2yv1kr&st=795s1alw&dl=0](https://www.dropbox.com/scl/fo/wxwa943qyeisgmz0sp2wv/AKEz-DIw8g8kI9bKRLwN9j8?rlkey=a6nghj80ogl7gzjb7re2yv1kr&st=795s1alw&dl=0)


## Acknowledgement
This study was carried out within the “AI-powered LIDAR fusion for next-generation smartphone cameras (LICAM)” project – funded by European Union – Next Generation EU within the PRIN 2022 program (D.D. 104 - 02/02/2022 Ministero dell’Università e della Ricerca). This manuscript reflects only the authors' views and opinions and the Ministry cannot be considered responsible for them.


## License 
Our code is released under MIT License (see LICENSE file for details).



