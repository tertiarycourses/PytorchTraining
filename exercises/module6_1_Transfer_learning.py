# Deep Learning with Pytorch
# Module 6: Neural Network
# Transfer Learning

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import requests

import torchvision.models as models
# model = models.resnet18(pretrained=True)
# model = models.alexnet(pretrained=True)
# model = models.squeezenet1_0(pretrained=True)
model = models.vgg16(pretrained=True)
# model = models.densenet_161(pretrained=True)
# model = models.inception_v3(pretrained=True)

image_path = 'images/football-299.jpg'

# Image pre-processing
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

img = Image.open(image_path)
img_tensor = preprocess(img)
img_tensor.unsqueeze_(0)
predict = model(Variable(img_tensor))

# Download ImageNet labels and store them as a dict
LABELS_URL = 'http://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

print(labels[predict.data.numpy().argmax()])