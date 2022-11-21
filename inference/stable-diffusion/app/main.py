import os
import gradio as gr
from huggingface_hub import login
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.diffusers.schedulers import GaudiDDIMScheduler


model_id = "CompVis/stable-diffusion-v1-4"
hf_hub_token = os.environ.get("HF_HUB_TOKEN", None)

if hf_hub_token is None:
    raise ValueError("Please set HF_HUB_TOKEN environment variable")
else:
    login(token=hf_hub_token)

# load model
scheduler = GaudiDDIMScheduler.from_config(model_id, subfolder="scheduler", use_auth_token=hf_hub_token)
generator = GaudiStableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    use_habana=True,
    use_lazy_mode=False,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)


def generate_image(prompt, guide, steps, num_images_per_prompt):
    outputs = generator(
        prompt=prompt,
        guidance_scale=guide,
        num_inference_steps=steps,
        num_images_per_prompt=num_images_per_prompt,
        batch_size=1,
    )
    return outputs


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

demo.queue(max_size=10, concurrency_count=8)

demo.launch(
    enable_queue=True,
    server_port=8080,
)
