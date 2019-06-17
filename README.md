# Grab AI Challenge: Computer Vision

## 1. Problem statement
Given a dataset of distinct car images, training a model to recognize the model and make of cars automatically.

### 1.1 Dataset overview
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.
website: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

## 2. Method description
The method is based on the WS-DAN (Weakly Supervised Data Augmentation Network) for FGVC (Fine-Grained Visual Classification). (Hu et al., ["See Better Before Looking Closer: Weakly Supervised Data Augmentation
Network for Fine-Grained Visual Classification"](https://arxiv.org/abs/1901.09891v2), arXiv:1901.09891)

Main idea of the method: Data augmentation is a good way to prevent overfitting and improve the performance of the deep-learning models. However, currently random data augmentation is used in most of the experiments and this method is inefficient and might introduce many uncontrolled background noises. WS-DAN tries to explore the potential of the data augmentation in two ways: see better and look closer. Seeing better means using attention maps represent discriminative parts of the objects and looking closer means localizing the object from the image and enlarging it to improve the model performance

## 3. Result
- In the paper, the author got an accuracy of 94.5% on Stanford Cars testing dataset.
- Following the ideas and implementation details mentioned in the paper, I got 94.04% on the same testing dataset. Here is the screenshot of the result and link. [http://imagenet.stanford.edu/internal/car196/submission/submission?p=27fe7e8d66d343cd755e19c3a4fc547c]

**Note: the email address I used to register Grab AI challenge and the email address I used to verify my anwser on ImageNet Server are the same**

## 4. Usage
This code repo contains WS-DAN with feature extractors including VGG19, ResNet, and Inception_v3 in PyTorch form. The default feature extractor is Resnet152, and this can be modified conveniently in ```train_wsdan.py```: 

```python
# feature_net = vgg19_bn(pretrained=True)
# feature_net = resnet101(pretrained=True)
feature_net = inception_v3(pretrained=True)

net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)
```

1. ``` git clone https://github.com/WangTianduo/Grab_Car_Classification.git ``` Use the command to clone the repo to local
2. Well-trained models are in the ```trained_mdoels``` folder
3. There is no need to prepare the dataset manually since the automatic downloading function has been integrated in the dataset.py
4. To train the model by yourself, simply run 
```$ python3 train_wsdan.py -j <num_workers> -b <batch_size> --sd <save_ckpt_directory> (etc.) 1>log.txt 2>&1 &``` (see ```train_wsdan.py``` for more training options) 
After running the script, the well_trained model will be stored in 	```trained_models``` folder. The name of the model file will the combination of feature extractor and the best accuracy
5. If you want to use the model that provided in the repo, use the method ```classifier()``` defined in ```test.py``` (A sample of how to use this method has also been provided)
