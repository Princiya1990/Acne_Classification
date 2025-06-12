import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse

# 1. Model: Generator architecture (from notebook)
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )
        self.final = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)
    def forward(self, x):
        block1 = self.block1(x)
        residual = self.residual_blocks(block1)
        block2 = self.block2(residual)
        out = block1 + block2
        out = self.upsample(out)
        out = self.final(out)
        return torch.tanh(out)

# 2. Utilities
def load_image(img_path, device):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),   # Adjust if needed
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def save_image(tensor, path):
    image = tensor.cpu().detach().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    image.save(path)

def super_resolve_images(generator, lr_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    for img_file in os.listdir(lr_dir):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        lr_img_path = os.path.join(lr_dir, img_file)
        lr_image = load_image(lr_img_path, device)
        with torch.no_grad():
            sr_image = generator(lr_image)
        output_path = os.path.join(output_dir, img_file)
        save_image(sr_image, output_path)
        print(f"Super-resolved {img_file}")

def show_result(lr_path, sr_path):
    lr_img = Image.open(lr_path).convert("RGB")
    sr_img = Image.open(sr_path).convert("RGB")
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('Low-Resolution')
    plt.imshow(lr_img)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Super-Resolved')
    plt.imshow(sr_img)
    plt.axis('off')
    plt.show()

# 3. Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_dir', type=str, required=True, help='Directory for low-res images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output super-resolved images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained generator .pth file')
    parser.add_argument('--show', type=str, default=None, help='(Optional) Show result for a file, just the filename (not path)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    generator = Generator()
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator = generator.to(device)
    generator.eval()

    super_resolve_images(generator, args.lr_dir, args.output_dir, device)

    # Optional: show side-by-side result
    if args.show:
        lr_path = os.path.join(args.lr_dir, args.show)
        sr_path = os.path.join(args.output_dir, args.show)
        show_result(lr_path, sr_path)
