import torch
from torch.nn import MaxPool2d
import timm


class my_resnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_model = timm.create_model(
            'resnet50', pretrained=True, num_classes=0, global_pool='avg')

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.act1(x)
        x = self.resnet_model.maxpool(x)

        x = self.resnet_model.layer1(x)
        # x = self.resnet_model.layer2(x)
        # x = self.resnet_model.layer3(x)

        x = self.resnet_model.forward_head(x)
        return x


def feature_extractor():
    '''
    This function will return a feature extractor of `resnet50`
    that will receive batch of image(s) in the format of tensor of shape `N x 3 x W X H`
    and output batch of feature(s) of shape `N x 2024`

    Return
    ------
    `torch nn.module`
    '''

    return my_resnet()
