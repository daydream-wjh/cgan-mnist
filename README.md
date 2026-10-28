# cGAN

条件生成对抗网络（Conditional GAN）生成 **MNIST 手写数字**：给定标签 0–9 与一个 100 维噪声向量，生成对应类别的 32×32 灰度数字图像。含训练脚本、Gradio 采样 demo，以及从第 0 轮到第 50 轮的检查点与按迭代轮次落盘的生成样例。

## 项目简介

- 训练脚本改编自 [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py)（文件首行注明出处），核心改动是加了损失记录、按 epoch 保存 checkpoint、训练结束画 G/D 损失曲线；
- `app.py` 用 Gradio 做交互式采样：下拉选一个类别，模型用一个随机 `torch.randn` 噪声生成该类别的一张 32×32 图；
- 仓库里的 `images/` 是训练过程中每 400 次迭代自动保存的 10×10 数字网格，`checkpoint_*.pth` 是可以继续训练/直接采样的完整训练状态；
- `myplot.png` / `myplot2023.png` 是 `app.py` 里把生成的第 2、0、2、3 号数字横向拼接得到的演示图（两张文件内容完全一致，MD5 相同），**不是损失曲线**。

## 效果展示

训练早期（第 400 次迭代）与后期（第 93600 次迭代）的生成网格，每行同一类别，左上到右下依次为 0–9 的采样：

![训练早期生成结果](./images/400.png)

![训练后期生成结果](./images/93600.png)

`app.py` 中 "2023" 四字拼接演示（同一噪声维度下按类别 2、0、2、3 生成后横向拼合）：

![生成 2023 演示](./myplot.png)

单张 32×32 生成结果示例：

![单张生成结果](./test.png)

## 数据集

**MNIST**（`torchvision.datasets.MNIST`，`train=True`，60000 张），预处理：

```python
transforms.Compose([
    transforms.Resize(opt.img_size),          # 28×28 → 32×32（单边整数 Resize）
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),       # 归一化到 [-1, 1]，与生成器末尾的 Tanh 输出区间匹配
])
```

- 类别数 `n_classes = 10`，图像尺寸 `img_size = 32`，通道 `channels = 1`（灰度）；
- 数据目录写死在 `../../data/mnist`（相对 `cgan.py` 所在位置，即仓库目录的**上一层**），首次运行 `download=True` 自动下载。这个路径来自上游项目 `implementations/cgan/` 的目录布局，直接在本仓库根目录运行会落到仓库外层的 `data/mnist`。

## 模型 / 算法

### 生成器 G

把 10 维标签嵌入与 100 维噪声**先拼后送**（`torch.cat((label_emb(labels), noise), -1)`，输入维度 110）：

| 层 | 输出 | 说明 |
| --- | --- | --- |
| `nn.Embedding(10, 10)` | 10 | 类别标签嵌入 |
| `Linear(110, 128)` | 128 | 首层不做 BatchNorm（`normalize=False`） |
| `Linear(128, 256)` + `BN1d(0.8)` + `LeakyReLU(0.2)` | 256 | |
| `Linear(256, 512)` + `BN1d(0.8)` + `LeakyReLU(0.2)` | 512 | |
| `Linear(512, 1024)` + `BN1d(0.8)` + `LeakyReLU(0.2)` | 1024 | |
| `Linear(1024, 1024)` | 32×32 | `int(np.prod(img_shape))`，随后 `view(-1, 1, 32, 32)` |
| `Tanh()` | | 输出区间 [-1, 1] |

全连接结构（非卷积 GAN），`BatchNorm1d` 的 momentum 固定为 0.8。

### 判别器 D

把展平的图像（1024 维）与标签嵌入（10 维）拼接成 1034 维输入：

```
Linear(1034, 1024) → LeakyReLU(0.2)
Linear(1024, 512)  → Dropout(0.4) → LeakyReLU(0.2)
Linear(512, 256)   → Dropout(0.4) → LeakyReLU(0.2)
Linear(256, 1)                          # 输出实值，不做 Sigmoid
```

### 训练策略

- 损失：`nn.MSELoss()`（LSGAN 风格，真样本目标 1.0、假样本目标 0.0，而非 BCE）；
- 优化器：两个 `Adam(lr=2e-4, betas=(0.5, 0.999))`，G / D 各一个；
- 每个 batch 先更新 G（用随机标签 `randint(0, 10)` 生成假样本骗 D），再更新 D（真样本 + `gen_imgs.detach()` 假样本，`d_loss = (d_real + d_fake) / 2`）；
- 默认超参：`n_epochs=50`、`batch_size=64`、`latent_dim=100`、`sample_interval=400`；
- 每 `sample_interval` 次迭代调用 `sample_image(n_row=10)`：采样 100 个噪声，标签按 `[num for _ in range(10) for num in range(10)]` 排列（每行同一类别），`save_image(..., nrow=10, normalize=True)` 存成 `images/<batches_done>.png`；
- 每个 batch 把 `d_loss.item()` / `g_loss.item()` 追加进 `d_losses` / `g_losses`（**按迭代而非按 epoch 记录**），最后一轮结束后用 matplotlib 画 G/D 曲线并 `plt.show()`。

### 评估指标

GAN 无监督，仓库里没有 FID/IS 等量化指标，也没有目视对比的评测脚本。唯一的"进度指标"是 `images/` 下按迭代轮次排列的网格图，以及训练时打印的 `D loss` / `G loss`。

## checkpoint 与 images 的对应关系

### `images/<数字>.png` 的命名

文件名 = `batches_done = epoch * len(dataloader) + i`，即**全局迭代次数（从 0 开始）**，每 400 次存一张：

- MNIST 60000 张 / batch 64 = 938 个 batch 每 epoch；
- 因此 `93600.png` ≈ `93600 / 938 ≈ 100` 个 epoch 末（说明这次实际跑了约 100 轮，而代码默认 `n_epochs=50`）；
- 仓库内共 **235 张**，从 `0.png` 到 `93600.png`，步长恰好 400；
- 每张是 10×10 = 100 个样本、342×342 像素（每个数字 32×32 加 margin），第 n 行全部是类别 n−1。

### `checkpoint_*.pth` 的结构

`checkpoint_{0,10,20,30,40,50}.pth` 都是 `torch.save` 的字典，已用 pickle 核对过键：

```python
{
  "epoch": int,                        # 注意是 0 基 epoch（checkpoint_50 里存的是 50）
  "generator_state_dict":     {...},   # 26 个键：label_emb.weight, model.0/2/3/5/6/8/9/11.*
  "discriminator_state_dict": {...},   # 9 个键：label_embedding.weight, model.0/1/2/3/4/6/7/8.*
  "optimizer_G_state_dict":   {"state": ..., "param_groups": ...},
  "optimizer_D_state_dict":   {"state": ..., "param_groups": ...},
}
```

- `generator.pth` / `generator1.pth` 是**裸的 `state_dict`**（26 个键，与 `generator_state_dict` 完全一致），适合"只要采样"的场景，每个约 6.7 MB；
- 完整 checkpoint 每个约 32 MB（`checkpoint_0.pth` 约 41 MB，体积异常，疑为另一次保存）；
- `app.py` 默认加载 `checkpoint_50.pth` 的 `generator_state_dict`。

### 用已有权重采样

```python
import torch
from torchvision.utils import save_image
from cgan import Generator          # 注意：先给 cgan.py 的模块级代码加 main 保护，见"已知问题"

g = Generator()
g.load_state_dict(torch.load("generator.pth", map_location="cpu"))  # 或 ckpt["generator_state_dict"]
g.eval()

z = torch.randn(100, 100)                       # latent_dim=100
labels = torch.arange(10).repeat_interleave(10) # 每行同一类别，0..9
with torch.no_grad():
    imgs = g(z, labels)
save_image(imgs, "sample.png", nrow=10, normalize=True)  # 32×32 灰度，[-1,1] 自动映射
```

### 从 checkpoint 继续训练

代码里**没有** resume 逻辑，需自行补上（在 `generator`/`discriminator`/两个 optimizer 初始化之后、训练循环之前）：

```python
ckpt = torch.load("checkpoint_50.pth", map_location="cpu")
generator.load_state_dict(ckpt["generator_state_dict"])
discriminator.load_state_dict(ckpt["discriminator_state_dict"])
optimizer_G.load_state_dict(ckpt["optimizer_G_state_dict"])
optimizer_D.load_state_dict(ckpt["optimizer_D_state_dict"])
start_epoch = ckpt["epoch"] + 1        # 注意存的是 0 基 epoch
```

然后让训练循环 `for epoch in range(start_epoch, opt.n_epochs)`。另外 `checkpoint_*.pth` 的命名与内部 `epoch` 字段对不上（`_50` 里存 `epoch=50`，而按代码 `f"checkpoint_{epoch+1}.pth"` 应存 51），说明这批文件来自不同版本的脚本或事后改过名，续训前请先 `print(ckpt["epoch"])` 确认。

## 技术栈与依赖

按代码 import 推断：

- Python 3（`f-string`、`os.makedirs` 用法，3.6+）
- `torch`、`torchvision`（`datasets.MNIST`、`save_image`、`make_grid`、`to_pil_image`）
- `numpy`、`matplotlib`
- `gradio`（**旧版 API**：`gr.inputs.Slider` / `gr.inputs.Dropdown` / `gr.outputs.Image` / `theme=` / `layout=`，gradio 4.x 已移除这些接口，需 `gradio<4`）

```bash
pip install torch torchvision numpy matplotlib "gradio<4"
```

（仓库无 `requirements.txt`。）

## 目录结构

```
cGAN/
├── cgan.py                    # G/D 定义 + 训练循环（模块级执行，改编自 PyTorch-GAN）
├── app.py                     # Gradio 采样 demo（加载 checkpoint_50.pth）
├── checkpoint_0.pth           # 完整训练状态（约 41 MB，体积异常）
├── checkpoint_10/20/30/40/50.pth   # 完整训练状态（各约 32 MB）
├── generator.pth              # 裸 generator state_dict（约 6.7 MB）
├── generator1.pth             # 同上，另一份（约 6.7 MB）
├── images/                    # 235 张按 batches_done 命名的 10×10 生成网格
│   ├── 0.png
│   ├── 400.png
│   └── … 93600.png
├── test.png                   # 单张 32×32 生成结果
├── myplot.png                 # app.py 的 "2023" 拼接演示图
└── myplot2023.png             # 与 myplot.png 内容相同
```

## 快速开始

### 1. 训练

```bash
cd cGAN
python cgan.py
```

MNIST 会下载到 `../../data/mnist`（仓库外层的 `data/mnist`）。训练中每 400 次迭代往 `images/` 写一张网格图，控制台逐 batch 打印 `[Epoch x/50] [Batch i/938] [D loss: …] [G loss: …]`，结束时弹出损失曲线窗口。

**注意**：`cgan.py` 用 `opt = parser.parse_args(args=[])` 固定了空参数列表，命令行传 `--n_epochs 100` 之类的参数**不会生效**，要改超参得直接改 `add_argument` 的 `default`。

### 2. 采样（Gradio 交互界面）

```bash
cd cGAN
python app.py
```

启动后浏览器里选类别（0–9），Gradio 用一个随机 `randn` 噪声生成该数字的 32×32 图像。启动前会先在终端 `plt.show()` 弹出 "2023" 演示图（见"已知问题"）。

### 3. 命令行批量采样

见上文「用已有权重采样」的片段，直接 `save_image` 出 10×10 网格，比走 Gradio 更省事。

## 已知问题

- **`cgan.py` 没有 `if __name__ == "__main__"` 保护，训练循环在模块级执行**：`app.py` 里的 `from cgan import Generator` 会连带把整段训练（含 50 epoch 默认设置、MNIST 下载、`plt.show()`）跑一遍。要单独采样，必须先把训练部分挪进 main 保护，或把 `Generator`/`Discriminator` 抽到独立模块。
- **checkpoint 实际不会被这份代码保存**：`if epoch+1 % 10 == 0:` 受运算符优先级影响等价于 `if (epoch + (1 % 10)) == 0`，对 `epoch >= 0` 恒为假；而 `torch.save(..., f"checkpoint_{epoch+1}.pth")` 又比内部 `'epoch': epoch` 大 1。仓库里的 6 个 checkpoint 应该来自别的版本或事后改名，请以 `ckpt["epoch"]` 为准。
- **命令行参数被忽略**：`parser.parse_args(args=[])` 强制空参数，`--n_epochs`、`--batch_size`、`--latent_dim`、`--sample_interval` 等都只能改默认值。
- **`images/` 显示跑了约 100 个 epoch**（最大 `93600.png` ÷ 938 batch/epoch），与默认 `n_epochs=50` 不一致；`checkpoint_0.pth` 体积也比其它检查点大 9 MB 左右。两者都说明仓库里的产物来自多次不同配置的运行，不能当作一次完整训练的线性记录。
- **`myplot.png` / `myplot2023.png` 不是损失曲线**：它们是 `app.py` 里类别 2、0、2、3 的拼接图，两份文件字节完全相同。真正的 G/D 曲线只 `plt.show()`，从未落盘。
- **`app.py` 细节问题**：`noise_input`（`gr.inputs.Slider`）定义了却没传给 `gr.Interface(fn=..., inputs=[class_input])`，界面上那个 Noise 滑杆实际无效；`plt.show()` 在 `gr.Interface(...).launch()` 之前执行会阻塞；`torch.load` 没有 `map_location`，CPU 机器加载 GPU 保存的权重会报错。
- **Gradio 旧 API**：`gr.inputs.*` / `gr.outputs.*` 在 gradio 4.x 被删除，只能装 `gradio<4` 才能跑。
- **路径硬编码**：数据目录 `../../data/mnist` 沿用上游多层目录布局，在本仓库根目录运行会把数据写到仓库外的父目录。
- **权重入库体积大**：6 个 checkpoint（约 200 MB）+ 2 个 `generator*.pth`（约 13 MB）+ 235 张生成图全部提交进 Git，未用 LFS / Release。
- **无 `requirements.txt`，也没有 `LICENSE` 文件**：原 README 只有一个 HuggingFace 风格的 `license: mit` 元数据块（无正文），本 README 已改写为正常文档。
- 生成质量：全连接 GAN、32×32、约 100 epoch，`images/93600.png` 里仍能看到部分数字（如 5、8、3）笔画粘连、同一行内形态发散，属于该结构在该训练量下的正常水平，没有可对比的量化指标。
