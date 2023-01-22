from diffusers import DiffusionPipeline
import gradio as gr
import torch
from PIL import Image
import time
import psutil
import random
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


start_time = time.time()
current_steps = 25
        
PIPE = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )


def inference(
    model_name,
    prompt,
    guidance,
    steps,
    n_images=1,
    width=512,
    height=512,
    seed=0,
    img=None,
    strength=0.5,
    neg_prompt="",
):

    print(psutil.virtual_memory())  # print memory usage

    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator("cuda").manual_seed(seed)

    try:
        return (
            img_to_img(
                model_name,
                prompt,
                n_images,
                neg_prompt,
                img,
                strength,
                guidance,
                steps,
                width,
                height,
                generator,
                seed,
            ),
            f"Done. Seed: {seed}",
        )
    except Exception as e:
        return None, error_str(e)


def img_to_img(
    model_name,
    prompt,
    n_images,
    neg_prompt,
    img,
    strength,
    guidance,
    steps,
    width,
    height,
    generator,
    seed,
):
    pipe = PIPE

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        num_images_per_prompt=n_images,
        image=img,
        num_inference_steps=int(steps),
        strength=strength,
        guidance_scale=guidance,
        generator=generator,
    )

    # return replace_nsfw_images(result)
    return result.images


def replace_nsfw_images(results):
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("nsfw.png")
    return results.images


with gr.Blocks(css="style.css") as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Protogen Diffusion</h1>
              </div>
              <p>
               Demo for multiple fine-tuned Protogen Stable Diffusion models.
              </p>
              <p>
               Running on <b>{device}</b>
              </p>
              <p>You can also duplicate this space and upgrade to gpu by going to settings:<br>
              <a style="display:inline-block" href="https://huggingface.co/spaces/patrickvonplaten/finetuned_diffusion?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p>
            </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=55):
            with gr.Group():
                model_name = gr.Dropdown(
                    label="Model",
                    choices=[m.name for m in models],
                    value=models[0].name,
                )
                with gr.Box(visible=False) as custom_model_group:
                    gr.HTML(
                        "<div><font size='2'>Custom models have to be downloaded first, so give it some time.</font></div>"
                    )

                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        show_label=False,
                        max_lines=2,
                        placeholder="Enter prompt.",
                    ).style(container=False)
                    generate = gr.Button(value="Generate").style(
                        rounded=(False, True, True, False)
                    )

                # image_out = gr.Image(height=512)
                gallery = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery"
                ).style(grid=[2], height="auto")

            state_info = gr.Textbox(label="State", show_label=False, max_lines=2).style(
                container=False
            )
            error_output = gr.Markdown()

        with gr.Column(scale=45):
            with gr.Tab("Options"):
                with gr.Group():
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="What to exclude from the image",
                    )

                    n_images = gr.Slider(
                        label="Images", value=1, minimum=1, maximum=4, step=1
                    )

                    with gr.Row():
                        guidance = gr.Slider(
                            label="Guidance scale", value=7.5, maximum=15
                        )
                        steps = gr.Slider(
                            label="Steps",
                            value=current_steps,
                            minimum=2,
                            maximum=75,
                            step=1,
                        )

                    seed = gr.Slider(
                        0, 2147483647, label="Seed (0 = random)", value=0, step=1
                    )

                with gr.Group():
                    image = gr.Image(
                        label="Image", height=256, tool="editor", type="pil"
                    )
                    strength = gr.Slider(
                        label="Transformation strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.5,
                    )

    inputs = [
        model_name,
        prompt,
        guidance,
        steps,
        n_images,
        width,
        height,
        seed,
        image,
        strength,
        neg_prompt,
    ]
    outputs = [gallery, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    ex = gr.Examples(
        [],
        inputs=[model_name, prompt, guidance, steps, neg_prompt],
        outputs=outputs,
        fn=inference,
        cache_examples=True,
    )

    gr.HTML(
        """
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>Models by <a href="https://huggingface.co/darkstorm2150">@darkstorm2150</a> and others. ❤️</p>
      <p>This space uses the <a href="https://github.com/LuChengTHU/dpm-solver">DPM-Solver++</a> sampler by <a href="https://arxiv.org/abs/2206.00927">Cheng Lu, et al.</a>.</p>
      <p>Space by: Darkstorm (Victor Espinoza)<br>
      <a href="https://www.instagram.com/officialvictorespinoza/">Instagram</a>
    </div>
    """
    )

print(f"Space built in {time.time() - start_time:.2f} seconds")

demo.queue(concurrency_count=1)
demo.launch()
