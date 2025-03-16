from collections import OrderedDict

from torch import nn
from torchvision import models

content_layers = {'block4_conv2': 0.5, 'block5_conv2': 0.5}
style_layers = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 'block3_conv1': 0.2, 'block4_conv1': 0.2, 'block5_conv1': 0.2}
def get_vgg19_model(layers):
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    model_with_convolution_layers = nn.Sequential(*list(model.children())[:36])

    for param in model_with_convolution_layers.parameters():
        param.requires_grad = False

    outputs = OrderedDict()
    for name, layer in zip(layers, model_with_convolution_layers.children()):
        outputs[name] = layer
        print(f"{name}: {layer}")
    return model, outputs

class ImageStyleTransfer(nn.Module):
    def __init__(self, cl=None, sl=None):
        super(ImageStyleTransfer, self).__init__()
        if sl is None:
            sl = style_layers
        if cl is None:
            cl = content_layers
        self.content_layers = cl
        self.style_layers = sl

        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        self.vgg19, self.outputs = get_vgg19_model(layers)

    def forward(self, x):
        content_outputs = []
        style_outputs = []

        for name, layer in self.outputs.items():
            x = layer(x)
            if name in self.content_layers:
                content_outputs.append((x, self.content_layers[name]))
            if name in self.style_layers:
                style_outputs.append((x, self.style_layers[name]))
        return {'content': content_outputs, 'style': style_outputs}

if __name__ == '__main__':
    ist_model = ImageStyleTransfer()
    print(ist_model)