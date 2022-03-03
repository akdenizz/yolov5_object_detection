# Object Detection with Yolov5

You can reach the base repo [here](https://github.com/ultralytics/yolov5)

* 🚀🌟 This repo can run any model that you trained before. 
* 🚀🌟 Put your .pt weights file in the folder configure the parameters in real_time_detection.py and RUN! 

## PyTorch Installation Guide

If you do not have conda environment, you should create with `conda create --name yolo` command in Anaconda Prompt.
Then activate it with `conda activate yolo` 

### Install PyTorch

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`

You can check any version which suitable for your computer [here](https://pytorch.org/get-started/locally/)

You should install Cuda version that satisfies your chosen above.


**After installation, check if you are on GPU or not**

```
python

>>>import torch

>>>torch.cuda.is_available()

>>>print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```

## After installation is completed, get this repo and install requirements

`git clone https://github.com/akdenizz/yolov5_object_detection`

`cd yolov5_object_detection`

`pip install -r requirements.txt`

`python real_time_detection.py`

* This repo can run any model that you trained before.
* Put your .pt weights file in the folder configure the parameters in real_time_detection.py and RUN!

## FOR CUSTOM TRAINING

*You can train your model in COLAB* 

You can check the [yolov5_training.ipynb](https://github.com/akdenizz/yolov5_object_detection/blob/main/yolov5_training.ipynb) file in this repo.

