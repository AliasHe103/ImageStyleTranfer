import torch
from torch import nn
from torchvision import models

style_layers = {0: "block1_conv1", 5: "block2_conv1",
                10: "block3_conv1", 19: "block4_conv1", 28: "block5_conv1"}
content_layers = {21: "block4_conv2", 30: "block5_conv2"}

class ImageStyleTransfer(nn.Module):
    def __init__(self):
        super(ImageStyleTransfer, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        # pretrained model
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # normalize
        x = (x - self.mean) / self.std

        content_features = {}
        style_features = {}

        for layer_id, layer in enumerate(self.features):
            x = layer(x) # necessary
            if layer_id in self.content_layers:
                content_features[self.content_layers[layer_id]] = x
            if layer_id in self.style_layers:
                style_features[self.style_layers[layer_id]] = x
            if layer_id >= max(self.content_layers.keys() | self.style_layers.keys()):
                break

        return content_features, style_features

if __name__ == '__main__':
    ist_model = ImageStyleTransfer()
    print(ist_model)