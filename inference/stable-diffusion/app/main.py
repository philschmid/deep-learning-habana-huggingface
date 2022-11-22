import os
import gradio as gr
from huggingface_hub import login
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.diffusers.schedulers import GaudiDDIMScheduler
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

# Import Habana Torch Library
import habana_frameworks.torch.core as htcore

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc1   = nn.Linear(784, 256)
        self.fc2   = nn.Linear(256, 64)
        self.fc3   = nn.Linear(64, 10)

    def forward(self, x):

        out = x.view(-1,28*28)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

# Target the Gaudi HPU device
device = torch.device("hpu")
net = SimpleModel()
net.to(device)




model_id = "CompVis/stable-diffusion-v1-4"
hf_hub_token = os.environ.get("HF_HUB_TOKEN", None)

if hf_hub_token is None:
    raise ValueError("Please set HF_HUB_TOKEN environment variable")
else:
    login(token=hf_hub_token)

# # load model
# scheduler = GaudiDDIMScheduler.from_config(model_id, subfolder="scheduler", use_auth_token=hf_hub_token)
# generator = GaudiStableDiffusionPipeline.from_pretrained(
#     model_id,
#     scheduler=scheduler,
#     use_habana=True,
#     use_lazy_mode=False,
#     use_hpu_graphs=True,
#     gaudi_config="Habana/stable-diffusion",
# )


def generate_image(prompt, guide, steps, num_images_per_prompt):
    outputs = generator(
        prompt=prompt,
        guidance_scale=guide,
        num_inference_steps=steps,
        num_images_per_prompt=num_images_per_prompt,
        batch_size=1,
    )
    return outputs.images


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt Input Text"),
        gr.Slider(2, 15, value=7, label="Guidence Scale"),
        gr.Slider(25, 75, value=50, step=1, label="Number of Iterations"),
        gr.Slider(label="Number of Images", minimum=1, maximum=16, step=1, value=1),
    ],
    outputs="image",
    title="Stable Diffusion on Habana Gaudi",
)

# demo.queue(max_size=10, concurrency_count=8)

# demo.launch(
#     enable_queue=True,
#     server_port=8080,
# )



app = Starlette(
    debug=True,
    routes=[],
    on_startup=[],
)
app = gr.mount_gradio_app(app, demo, path="/")

# HF_HUB_TOKEN=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 python3 -m uvicorn app.main:app  --workers 2

# HF_HUB_TOKEN=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 python3 app/main.py 