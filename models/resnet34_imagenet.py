"""ResNet-34 for ImageNet-1k: torchvision's pretrained layout, 1000 classes"""
from ._resnet34 import ResNet34


class Model(ResNet34):
    def __init__(self):
        super().__init__(n_classes=1000)
