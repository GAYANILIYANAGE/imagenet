# ImageNet training in PyTorch
This trains common model architectures such as ResNet, AlexNet, and VGG on the ImageNet dataset.

## Requirements
Install PyTorch (pytorch.org)
pip install -r requirements.txt
Download the ImageNet dataset from [http://www.image-net.org/](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)

##Training
To train a model, run imagenet.py with the desired model architecture and the path to the ImageNet dataset
python main.py -a resnet18 [imagenet-folder with train and val folders]
