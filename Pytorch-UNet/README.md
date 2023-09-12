# U-Net: Semantic segmentation with PyTorch
The repo is a fork of [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). Detailed instruction for dependencies and installation can be found there. 
## Creating new labels
Masking images can be automatically created using command
```
python3 ./data/process_mask.py
```
Masks can be created from RGB images with clear segmentation as shown below.

![Segmented Image](data/masks/img_orig_2500.jpg)

The red region can be converted to masks as 

![Mask Image](data/label/img_orig_2500.png)

## Training using existing data
The training data are located in folder ``data/imgs`` and the training masks are located in folder ``data/label``. To train a new model, run command 
```
python3 train.py
```
in the root folder of the repo. Detailed command for how the script works can be found by using command 
```
python3 train.py -h
```

## Feature point extraction using existing model
To test the obtained model, run command 
```
python3 predict.py -m $(checkpoint filename.pth) -i $(input image filename) -o $(output image filename)
```
Currently predict.py is written to find the 4 corner of the strips at the beginning of the run way. An example result is shown below.

![Keypoint Image](Figure_1.png)

The four keypoints detected are top left corner (red), bottom left corner (green), bottom right corner (blue), top right corner (yellow). 
