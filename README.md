# MesoNet-Pytorch
-------------------------------------------
The is a personal Reimplemention of MesoNet[1] using Pytorch. If you make use of this work, please cite the paper accordingly. 

For the original version of this work using Keras, please see: [DariusAf/MesoNet](https://github.com/DariusAf/MesoNet)

## Install & Requirements
The code has been test on pytorch 1.3.1, torchvision 0.4.2 and python 3.6.9, please refer to `requirements.txt` for more details.

**To install the python packges**

`python -m pip install -r requiremnets.txt`

## Usage
**To train the normal MesoNet**

`python train_Meso.py -n 'Mesonet' -tp './data/train' -vp './data/val' -bz 64 -e 100 -mn 'meso4.pkl'`

**To train the MesoInceptionNet**

`python train_MesoInception.py -n 'MesoInception' -tp './data/train' -vp './data/val' -bz 64 -e 100 -mn 'mesoinception.pkl'`


If you continue training a pretrained model, you should use `--continue_train True -mp ./pretrained_models/model.pkl`

**To test the trained Model**

`python test.py -bz 64 -tp './data/test' -mp './Mesonet/best.pkl'`

## License
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.

## Reference
[1] Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018, September). MesoNet: a Compact Facial Video Forgery Detection Network. In IEEE Workshop on Information Forensics and Security, WIFS 2018.
