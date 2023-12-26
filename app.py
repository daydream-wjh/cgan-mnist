import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from cgan import Generator

latent_dim = 100
n_classes = 10
img_size = 32
channels = 1
checkpoint = torch.load("checkpoint_50.pth")
model = Generator()
model.load_state_dict(checkpoint['generator_state_dict'])
model.eval()


# 生成单个数字的图片
def generate_image(class_idx):
    with torch.no_grad():
        # Generate random noise vector of latent_dim size
        noise = torch.randn(1, latent_dim)
        label = torch.tensor([int(class_idx)])
        gen_img = model(noise, label).squeeze(0)
    return to_pil_image(make_grid(gen_img, normalize=True))


# Generate each digit
image2 = generate_image(2)
image0 = generate_image(0)
image2_2 = generate_image(2)
image3 = generate_image(3)

# Convert PIL Images to numpy arrays
image2 = np.array(image2)
image0 = np.array(image0)
image2_2 = np.array(image2_2)
image3 = np.array(image3)

# Concatenate images along the width
image_2023 = np.concatenate((image2, image0, image2_2, image3), axis=1)

# Convert back to PIL Image and show
image_2023 = Image.fromarray(image_2023)
plt.imshow(image_2023, cmap='gray')
plt.show()

# Create Gradio Interface
noise_input = gr.inputs.Slider(minimum=-1.0, maximum=1.0, default=0, step=0.1, label="Noise")
class_input = gr.inputs.Dropdown([str(i) for i in range(n_classes)], label="Class")
output_image = gr.outputs.Image('pil')



gr.Interface(
    fn=generate_image,
    inputs=[class_input],
    outputs=output_image,
    title="MNIST Generator",
    description="Generate images of handwritten digits from the MNIST dataset using a GAN.",
    theme="default",
    layout="vertical",
    live=True
).launch(debug=True)
