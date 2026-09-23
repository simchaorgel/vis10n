"""Copies torchvision's pretrained ResNet-34 weights into our ResNet34 and saves them.

No training involved: torchvision's weights are copied across by position, which works
because our layers are declared in the same order. Run it from the repo root with

    python -m training.copy_pretrained_imagenet

then import the file it writes through the app's load wizard:
architecture resnet34_imagenet, predicts ImageNet, accuracy 73.3.
"""
import torch as t
from torchvision import models
from torchvision.models import ResNet34_Weights

from models.resnet34_imagenet import Model

OUT = "imagenet_resnet34.pt"


def copy_weights(mine: t.nn.Module, pretrained: t.nn.Module) -> t.nn.Module:
    """Copies parameters and buffers across by position, checking the shapes line up."""
    mydict, theirdict = mine.state_dict(), pretrained.state_dict()
    assert len(mydict) == len(theirdict), "Mismatching state dictionaries."

    state_dict_to_load = {}
    for (mykey, myvalue), (theirkey, theirvalue) in zip(mydict.items(), theirdict.items()):
        assert myvalue.shape == theirvalue.shape, f"{mykey} != {theirkey}"
        state_dict_to_load[mykey] = theirvalue

    mine.load_state_dict(state_dict_to_load)
    return mine


if __name__ == "__main__":
    pretrained = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model = copy_weights(Model(), pretrained).eval()

    # Same input through both should give the same answer, bar floating point noise
    x = t.randn(1, 3, 224, 224)
    with t.inference_mode():
        difference = (model(x).softmax(1) - pretrained.eval()(x).softmax(1)).abs().max().item()
    print(f"largest difference from torchvision: {difference:.2e}")

    t.save(model.state_dict(), OUT)
    print(f"saved {OUT} - import it with architecture resnet34_imagenet, predicts ImageNet")
