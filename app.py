from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import gradio as gr
import torch
from PIL import Image
import time
import psutil
import random
# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


start_time = time.time()
current_steps = 15

pipe = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

if torch.cuda.is_available():
    pipe = pipe.to("cuda")


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )


def inference(
    prompt,
    text_guidance_scale,
    image_guidance_scale,
    image,
    steps,
    neg_prompt="",
    width=512,
    height=512,
    seed=0,
):
    print(psutil.virtual_memory())  # print memory usage

    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator("cuda").manual_seed(seed)

    try:
        ratio = min(height / image.height, width / image.width)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            image=image,
            num_inference_steps=int(steps),
            image_guidance_scale=image_guidance_scale,
            guidance_scale=text_guidance_scale,
            generator=generator,
        )

        # return replace_nsfw_images(result)
        return result.images, f"Done. Seed: {seed}"
    except Exception as e:
        return None, error_str(e)


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
                <h1>Instruct-Pix2Pix Diffusion</h1>
              </div>
              <p>
               Demo for Instruct-Pix2Pix Diffusion
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
                        steps = gr.Slider(
                            label="Steps",
                            value=current_steps,
                            minimum=2,
                            maximum=75,
                            step=1,
                        )

                    with gr.Row():
                        width = gr.Slider(
                            label="Width", value=512, minimum=64, maximum=1024, step=8
                        )
                        height = gr.Slider(
                            label="Height", value=512, minimum=64, maximum=1024, step=8
                        )

                    seed = gr.Slider(
                        0, 2147483647, label="Seed (0 = random)", value=0, step=1
                    )

                with gr.Group():
                    image = gr.Image(
                        label="Image", height=256, tool="editor", type="pil"
                    )
                    text_guidance_scale = gr.Slider(
                        label="Text Guidance Scale", minimum=1.0, value=5.5, maximum=15, step=0.1
                    )
                    image_guidance_scale = gr.Slider(
                        label="Image Guidance Scale",
                        minimum=1.0,
                        maximum=15,
                        step=0.1,
                        value=1.5,
                    )

    inputs = [
        prompt,
        text_guidance_scale,
        image_guidance_scale,
        image,
        steps,
        neg_prompt,
        width,
        height,
        seed,
    ]
    outputs = [gallery, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    ex = gr.Examples(
        [
            ["turn him into a cyborg", 7.5, 1.2, "./statue.jpg", 20]
        ],
        inputs=[prompt, text_guidance_scale, image_guidance_scale, image, steps],
        outputs=outputs,
        fn=inference,
        cache_examples=True,
    )

print(f"Space built in {time.time() - start_time:.2f} seconds")

demo.queue(concurrency_count=1)
demo.launch()
