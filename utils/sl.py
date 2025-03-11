import torch
from PIL import Image
from torchvision import transforms

image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.299, 0.224, 0.225])

# width and height of the target image
scale = 16 / 9
target_width = 512
target_height = int(512 / scale)
def load_image(image_path, width=target_width, height=target_height):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    return image

def save_image(image, path):
    # [C, H, W]
    image = image.squeeze(0)
    image = image * image_std.view(3, 1, 1) + image_mean.view(3, 1, 1)

    image = image.clamp(0, 1) * 255.0

    # to PIL
    image = image.byte()
    image = transforms.ToPILImage()(image)
    image.save(path)
