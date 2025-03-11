import os
import torch
from tqdm import tqdm
from utils.sl import load_image, save_image, target_height, target_width
from src.model import ImageStyleTransfer

# input and output path
CONTENT_IMAGE_PATH = './input/content.jpg'
STYLE_IMAGE_PATH = './input/style.jpg'
OUTPUT_DIR = './output'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageStyleTransfer()
content_image = load_image(CONTENT_IMAGE_PATH)
style_image = load_image(STYLE_IMAGE_PATH)
# print(content_image)
# print(style_image)

# forward and get features
target_content_features = model(content_image)['content']
target_style_features = model(style_image)['style']
total_pixels = target_width * target_height
channels = 3

def content_loss(features):
    _losses = []
    for (feature, factor), (target_feature, _) in zip(features, target_content_features):
        _loss = torch.sum((feature - target_feature) ** 2) # MSE
        _loss = _loss / (channels * total_pixels * 2.0) # Normalization
        _losses.append(_loss * factor)

    return torch.sum(torch.stack(_losses))

def gram_matrix(feature):
    x = feature.view(feature.size(1), -1) # (C, H * W)
    return torch.mm(x, x.t())
def style_loss(features):
    _losses = []
    for (feature, factor), (target_feature, _) in zip(features, target_style_features):
        _loss = gram_matrix(feature) - gram_matrix(target_feature)
        _loss = torch.sum(_loss ** 2)
        _loss = _loss / ((total_pixels ** 2) * (channels ** 2) * 4.0)
        _losses.append(_loss * factor)

    return torch.sum(torch.stack(_losses))

def total_loss(features):
    # layers and factors
    # 1: 100
    w1, w2 = 1, 100
    loss_of_content = content_loss(features['content'])
    loss_of_style = style_loss(features['style'])
    _loss = loss_of_content * w1 + loss_of_style * w2

    return _loss


# epochs, steps per epoch, learning rate
epochs = 10
steps = 100
lr = 0.01

noise_image = torch.clone(content_image).detach().requires_grad_(True)
optimizer = torch.optim.Adam([noise_image], lr=lr)
def train_by_step():
    optimizer.zero_grad()
    noise_outputs = model(noise_image)
    _loss = total_loss(noise_outputs)
    _loss.backward()
    optimizer.step()

    return _loss

def train():
    for epoch in range(epochs):
        with tqdm(total=steps, desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
            for step in range(steps):
                loss = train_by_step()
                pbar.set_postfix({'loss': '%.4f' % (loss.item())})
                pbar.update()
            save_image(noise_image, os.path.join(OUTPUT_DIR, f'{epoch + 1}.jpg'))

if __name__ == '__main__':
    train()
