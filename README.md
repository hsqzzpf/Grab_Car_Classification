# Grab AI Challenge: Computer Vision

## 1. Problem statement
Given a dataset of distinct car images, training a model to recognize the model and make of cars automatically.

### 1.1 Dataset overview
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

website: [https://ai.stanford.edu/~jkrause/cars/car_dataset.html]

## 2. Method description

Main idea of the method: Data augmentation is a good way to prevent overfitting and improve the performance of the deep-learning models. However, currently random data augmentation is used in most of the experiments and this method is inefficient and might introduce many uncontrolled background noises. The method tries to explore the potential of the data augmentation in two ways: see better and look closer. Seeing better means using attention maps represent discriminative parts of the objects and looking closer means localizing the object from the image and enlarging it to improve the model performance

## 3. Result

- By implementing the ideas described in above, I got 94.04% on the testing dataset. Here is the screenshot of the result and link. [http://imagenet.stanford.edu/internal/car196/submission/submission?p=27fe7e8d66d343cd755e19c3a4fc547c]

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
3. Use ``` pip3 install -r requirement.txt```  install all the required packages in the conda environment
4. There is no need to prepare the dataset manually since the automatic downloading function has been integrated in the dataset.py
5. To train the model by yourself, simply run 
```$ python3 train_wsdan.py ``` (see ```train_wsdan.py``` for more training options) 
After running the script, the well_trained model will be stored in 	```trained_models``` folder. The name of the model file will the combination of feature extractor and the best accuracy
6. If you want to use the model that provided in the repo, use the method ```classifier()``` defined in ```test.py``` (A sample of how to use this method has also been provided)

**Note: If it crashes when using the requirement.txt, try to download these packages manually **
- torch == 1.1.0
- torchvision == 0.2.2
- Pillow == 6.0.0
- scipy == 1.2.1
- numpy == 1.16.3

## 5. Reference
[1] Hu et al., See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification. In CVPR, 2019.

[2] H.Zheng et al., Learning multi-attention convolutional neural network for fine-grained image recognition. In ICCV, 2017.

[3] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[4] J. Fu et al., Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition. In CVPR, 2017.