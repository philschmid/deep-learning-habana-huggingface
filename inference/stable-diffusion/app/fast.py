from starlette.applications import Starlette
import gradio as gr
import os 
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

print(os.getpid())

def read_main(req):
    return {"message": "This is your main app"}

io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")


async def some_startup_task():
    print("test")


app = Starlette(
    debug=True,
    routes=[
        Route("/", read_main, methods=["GET"]),
    ],
    on_startup=[some_startup_task],
)
app = gr.mount_gradio_app(app, io, path="/gradio")

# python3 -m uvicorn app.fast:app  --workers 2