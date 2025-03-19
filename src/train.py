import os

import torch
from PIL import Image
from torchvision import transforms, utils
from tqdm import tqdm

from src.model import ImageStyleTransfer

epochs = 10
steps_per_epoch = 100
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor()
])

OUTPUT_DIR = './output'
content_image = Image.open("./input/content.jpg")
style_image = Image.open("./input/style.jpg")
content_tensor = loader(content_image).unsqueeze(0).to(device)
style_tensor   = loader(style_image).unsqueeze(0).to(device)

model = ImageStyleTransfer().to(device)
content_targets, style_targets = model(content_tensor), model(style_tensor)

def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    # shape: [batch_size, channels, height, width]
    b, c, h, w = features.size()
    F = features.view(c, h * w)
    # gram
    G = torch.mm(F, F.t())
    return G / torch.mean(G)  # 取均值归一化

style_target_grams = {}
for layer, feat_map in style_targets[1].items():
    style_target_grams[layer] = gram_matrix(feat_map)

result_image = content_tensor.clone().to(device)
result_image.requires_grad_(True)

optimizer = torch.optim.Adam([result_image], lr=learning_rate)

def compute_content_loss(features):
    content_loss = 0.0
    for layer, target_feat in content_targets[0].items():
        layer_feature = features[layer]
        content_loss += torch.nn.functional.mse_loss(layer_feature, target_feat)
    return content_loss

def compute_style_loss(features):
    style_loss = 0.0
    for layer, target_gram in style_target_grams.items():
        gram = gram_matrix(features[layer])
        style_loss += torch.nn.functional.mse_loss(gram, target_gram)
    return style_loss

w1, w2 = 1.0, 100.0
def train_step():
    optimizer.zero_grad()
    gen_content_feats, gen_style_feats = model(result_image)
    total_loss = w1 * compute_content_loss(gen_content_feats) + w2 * compute_style_loss(gen_style_feats)
    total_loss.backward()
    optimizer.step()
    with torch.no_grad():
        result_image.clamp_(0.0, 1.0)
    return total_loss

def train():
    for epoch in range(epochs):
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for step in range(steps_per_epoch):
                loss = train_step()
                pbar.set_postfix({'loss': '%.4f' % (loss.item())})
                pbar.update()
            output_image = result_image.detach().cpu()  # 将张量移到CPU
            utils.save_image(output_image, os.path.join(OUTPUT_DIR, f'{epoch + 1}.jpg'))


if __name__ == '__main__':
    train()
