"""ResNet-34 for CIFAR-10: ImageNet stem, 10 classes"""
from ._resnet34 import ResNet34

class Model(ResNet34):
    def __init__(self):
        super().__init__(n_classes=10)
