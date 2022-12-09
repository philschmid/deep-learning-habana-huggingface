import os
os.system("hl-smi")

import gradio as gr
from huggingface_hub import login
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.diffusers.schedulers import GaudiDDIMScheduler
from starlette.applications import Starlette

app = Starlette(
    debug=True,
    routes=[],
    on_startup=[],
)

model_id = "runwayml/stable-diffusion-v1-5"
hf_hub_token = os.environ.get("HF_HUB_TOKEN", None)

if hf_hub_token is not None:
    login(token=hf_hub_token)

# load model
scheduler = GaudiDDIMScheduler.from_config(model_id, subfolder="scheduler", use_auth_token=hf_hub_token)
pipe = GaudiStableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    use_habana=True,
    use_lazy_mode=False,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
pipe.safety_checker = None

def infer(prompt, guide=7, steps=50, num_images_per_prompt=4):
    outputs = pipe(
        prompt=prompt,
        guidance_scale=guide,
        num_inference_steps=steps,
        num_images_per_prompt=num_images_per_prompt,
        batch_size=4,
    )
    return outputs.images

# runs first generation for fast integration speed
# infer("test shield")

css = """
        a {
            color: inherit;
            text-decoration: underline;
        }
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: #0370C1;
            background: #0370C1;
        }
        .dark .gr-button {
            color: white;
            border-color: #0370C1;
            background: #0370C1;
        }
        .gr-box {
            font-size: .875rem;
            color: #000;
        }
        .gr-form{
            flex: 1 1 50%; 
            border-top-right-radius: 0; 
            border-bottom-right-radius: 0;
        }
        input[type='range'] {
            accent-color: #0370C1;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-options {
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        # .dark .logo{ filter: invert(1); }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
"""

block = gr.Blocks(css=css)

examples = [
    [
       "steampunk market interior, colorful, 3 d scene, greg rutkowski, zabrocki, karlkka, jayison devadas, trending on artstation, 8 k, ultra wide angle, zenith view, pincushion lens effect",
    ],
    [
"A portrait of a cyborg in a golden suit, D&D sci-fi, artstation, concept art, highly detailed illustration."
    ],
    [
        "ultra realistic illustration of taco cat, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
    ],
    ["rubik's cube transformer, tristan eaton, victo ngai, artgerm, rhads, ross draws"],
    
]

with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto; display:grid; gap:25px;">
                <img class="logo" src="https://huggingface.co/datasets/philschmid/assets/resolve/main/habana_hf.png" alt="Hugging Face Habana Logo"
                    style="margin: auto; max-width: 14rem;">
                <h1 style="font-weight: 900; font-size: 3rem; line-height: 3rem;">
                  Stable Diffusion on Habana Gaudi 
                </h1>
              <p style="margin-bottom: 10px; font-size: 94%">
              Generate new Images from a text description,
                <a href="https://www.philschmid.de">created by Hugging Face and Habana</a>.
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")


        ex = gr.Examples(examples=examples, fn=infer, inputs=[text], outputs=gallery, cache_examples=False)
        ex.dataset.headers = [""]

        text.submit(infer, inputs=[text], outputs=gallery)
        btn.click(infer, inputs=[text], outputs=gallery)
        gr.HTML(
            """
                <div class="footer">
                    <p> Gradio Demo by 🤗 Hugging Face and Habana Labs
                    </p>
                </div>
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
           """
        )
        
# Single HPU
block.queue(concurrency_count=1,max_size=10)
block.launch(server_name="0.0.0.0",server_port=8000)

# Multi HPU
# start with CMD ["python3", "-m", "uvicorn", "main:app", "--workers", "2"]
# app = gr.mount_gradio_app(app, block, path="/")

