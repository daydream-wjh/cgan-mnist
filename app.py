"""cGAN 采样界面：按类别生成 MNIST 手写数字。

用法：
    python app.py                        # 默认加载同目录下 checkpoint_50.pth
    python app.py --ckpt checkpoint_10.pth
    python app.py --share                # 生成 gradio 公网临时链接
"""
import argparse
import os

import gradio as gr
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from cgan import Generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LATENT_DIM = 100
N_CLASSES = 10


def load_model(ckpt_path, device):
    """加载 checkpoint 里的生成器权重。"""
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到权重文件 {ckpt_path}。请先运行 `python cgan.py` 训练，或用 --ckpt 指定别的路径。"
        )

    try:
        # PyTorch >= 2.6 默认 weights_only=True，这里存的是 state_dict，两者都兼容
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # 老版本 torch 没有 weights_only 参数
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = Generator().to(device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    return model


def build_interface(model, device):
    def generate_image(class_idx, n_row):
        """按类别生成 n_row×n_row 个样本，拼成一张网格图。"""
        class_idx = int(class_idx)
        n_row = int(n_row)
        with torch.no_grad():
            noise = torch.randn(n_row * n_row, LATENT_DIM, device=device)
            labels = torch.full((n_row * n_row,), class_idx, dtype=torch.long, device=device)
            gen_img = model(noise, labels).cpu()
        grid = make_grid(gen_img, nrow=n_row, normalize=True)
        return to_pil_image(grid)

    return gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Dropdown(choices=[str(i) for i in range(N_CLASSES)], value="0", label="数字类别"),
            gr.Slider(minimum=1, maximum=5, value=3, step=1, label="网格大小 n×n"),
        ],
        outputs=gr.Image(type="pil", label="生成结果"),
        title="MNIST Generator",
        description="用条件 GAN 按类别生成 MNIST 手写数字。每次点击都会重新采样，结果都不同。",
        allow_flagging="never",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoint_50.pth", help="checkpoint 路径")
    parser.add_argument("--share", action="store_true", help="生成 gradio 公网临时链接")
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    print(f"已在 {device} 上加载 {args.ckpt}")

    demo = build_interface(model, device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
