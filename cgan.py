"original code: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py"
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 所有相对路径都以本文件所在目录为基准，避免「换个目录启动就找不到数据/权重」
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
# 训练轮数
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--ckpt_interval", type=int, default=10, help="每多少轮存一次 checkpoint")
parser.add_argument("--data_dir", type=str, default=os.path.join(BASE_DIR, "data", "mnist"),
                    help="MNIST 存放目录")
parser.add_argument("--out_dir", type=str, default=os.path.join(BASE_DIR, "images"),
                    help="采样图与 checkpoint 输出目录")
# 用 parse_known_args：这样 app.py 被 gradio 带参数启动时，也不会因为陌生参数直接崩
opt, _unknown_args = parser.parse_known_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)


def get_device():
    """有 GPU 用 GPU，没有就退回 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


def sample_image(generator, device, out_dir, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(n_row ** 2, opt.latent_dim, device=device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = torch.tensor([num for _ in range(n_row) for num in range(n_row)], device=device)
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, os.path.join(out_dir, "%d.png" % batches_done),
               nrow=n_row, normalize=True)


def save_checkpoint(path, epoch, generator, discriminator, optimizer_G, optimizer_D):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, path)


def main():
    print(opt)

    device = get_device()
    cuda = device.type == "cuda"

    os.makedirs(opt.out_dir, exist_ok=True)
    os.makedirs(opt.data_dir, exist_ok=True)

    # Loss functions（LSGAN 式的最小二乘损失）
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Configure data loader
    dataloader = DataLoader(
        datasets.MNIST(
            opt.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu if cuda else 0,
        pin_memory=cuda,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------
    d_losses = []
    g_losses = []

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.full((batch_size, 1), 1.0, device=device)
            fake = torch.full((batch_size, 1), 0.0, device=device)

            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.long)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, opt.latent_dim, device=device)
            gen_labels = torch.randint(0, opt.n_classes, (batch_size,), device=device)

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch + 1, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(generator, device, opt.out_dir, n_row=10, batches_done=batches_done)

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        # 原代码写成 `if epoch+1 % 10 == 0`，因为 % 比 + 优先级高，
        # 实际等价于 `if epoch + 1`，条件恒为真 —— 于是每一轮都会存一次 checkpoint。
        # 这里补上括号，恢复「每 ckpt_interval 轮存一次」的原意。
        if (epoch + 1) % opt.ckpt_interval == 0:
            save_checkpoint(os.path.join(opt.out_dir, f"checkpoint_{epoch + 1}.pth"),
                            epoch, generator, discriminator, optimizer_G, optimizer_D)

    # 最后一轮一定存一次，避免 n_epochs 不是 ckpt_interval 的整数倍时没有权重可用
    if opt.n_epochs % opt.ckpt_interval != 0:
        save_checkpoint(os.path.join(opt.out_dir, f"checkpoint_{opt.n_epochs}.pth"),
                        opt.n_epochs - 1, generator, discriminator, optimizer_G, optimizer_D)

    # 训练结束后画损失曲线：存成文件，不再用阻塞式的 plt.show()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    curve_path = os.path.join(opt.out_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"损失曲线已保存到 {curve_path}")


if __name__ == "__main__":
    main()
