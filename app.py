from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
import gradio as gr
import torch
from PIL import Image
import time
import psutil
import random
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


start_time = time.time()
current_steps = 25

SAFETY_CHECKER = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker", torch_dtype=torch.float16)


class Model:
    def __init__(self, name, path=""):
        self.name = name
        self.path = path

        if path != "":
            self.pipe_t2i = StableDiffusionPipeline.from_pretrained(
                path, torch_dtype=torch.float16, safety_checker=SAFETY_CHECKER
            )
            self.pipe_t2i.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe_t2i.scheduler.config
            )
            self.pipe_i2i = StableDiffusionImg2ImgPipeline(**self.pipe_t2i.components)
        else:
            self.pipe_t2i = None
            self.pipe_i2i = None


models = [
    Model("Protogen v2.2 (Anime)", "darkstorm2150/Protogen_v2.2_Official_Release"),
    Model("Protogen x3.4 (Photorealism)", "darkstorm2150/Protogen_x3.4_Official_Release"),
    Model("Protogen x5.3 (Photorealism)", "darkstorm2150/Protogen_x5.3_Official_Release"),
    Model("Protogen x5.8 Rebuilt (Scifi+Anime)", "darkstorm2150/Protogen_x5.8_Official_Release"),
    Model("Protogen Dragon (RPG Model)", "darkstorm2150/Protogen_Dragon_Official_Release"),
    Model("Protogen Nova", "darkstorm2150/Protogen_Nova_Official_Release"),
    Model("Protogen Eclipse", "darkstorm2150/Protogen_Eclipse_Official_Release"),
    Model("Protogen Infinity", "darkstorm2150/Protogen_Infinity_Official_Release"),
]

MODELS = {m.name: m for m in models}

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
        if img is not None:
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
        else:
            return (
                txt_to_img(
                    model_name,
                    prompt,
                    n_images,
                    neg_prompt,
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


def txt_to_img(
    model_name,
    prompt,
    n_images,
    neg_prompt,
    guidance,
    steps,
    width,
    height,
    generator,
    seed,
):
    pipe = MODELS[model_name].pipe_t2i

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        num_images_per_prompt=n_images,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )

    pipe.to("cpu")
    torch.cuda.empty_cache()

    return replace_nsfw_images(result)


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
    pipe = MODELS[model_name].pipe_i2i

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

    pipe.to("cpu")
    torch.cuda.empty_cache()

    return replace_nsfw_images(result)


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
                    custom_model_path = gr.Textbox(
                        label="Custom model path",
                        placeholder="Path to model, e.g. darkstorm2150/Protogen_x3.4_Official_Release",
                        interactive=True,
                    )
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

            with gr.Tab("Image to image"):
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
        [
            [models[0].name, "portrait of a beautiful alyx vance half life", 10, 50, "canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
            [models[1].name, "Brad Pitt with sunglasses, highly realistic", 7.5, 50, "canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
            [models[2].name, "(extremely detailed CG unity 8k wallpaper), the most beautiful artwork in the world", 7.5, 50, "human, people, canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
            [models[3].name, "(extremely detailed CG unity 8k wallpaper), full shot body photo star lord chris pratt posing in an outdoor spaceship, holding a gun, extremely detailed, trending on ArtStation, trending on CGSociety, Intricate, High Detail, dramatic, realism, beautiful and detailed lighting, shadows", 7.5, 50, "canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
            [models[4].name, "(extremely detailed CG unity 8k wallpaper), full body portrait of (david:1.1), staring at us with a mysterious gaze, realistic, masterpiece, highest quality, ((scifi)), lens flare, ((light sparkles)), unreal engine, digital painting, trending on ArtStation, trending on CGSociety, Intricate, High Detail, dramatic, realism, beautiful and detailed lighting, shadows", 7.5, 50, "canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
            [models[5].name, "(extremely detailed CG unity 8k wallpaper), full body portrait of (david:1.1), staring at us with a mysterious gaze, realistic, masterpiece, highest quality, ((scifi)), lens flare, ((light sparkles)), unreal engine, digital painting, trending on ArtStation, trending on CGSociety, Intricate, High Detail, dramatic, realism, beautiful and detailed lighting, shadows", 7.5, 50, "canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
            [models[6].name, "(extremely detailed CG unity 8k wallpaper), full body portrait of (david:1.1), staring at us with a mysterious gaze, realistic, masterpiece, highest quality, ((scifi)), lens flare, ((light sparkles)), unreal engine, digital painting, trending on ArtStation, trending on CGSociety, Intricate, High Detail, dramatic, realism, beautiful and detailed lighting, shadows", 7.5, 50, "canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
            [models[7].name, "(extremely detailed CG unity 8k wallpaper), full body portrait of (david:1.1), staring at us with a mysterious gaze, realistic, masterpiece, highest quality, ((scifi)), lens flare, ((light sparkles)), unreal engine, digital painting, trending on ArtStation, trending on CGSociety, Intricate, High Detail, dramatic, realism, beautiful and detailed lighting, shadows", 7.5, 50, "canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy"],
        ],
        inputs=[model_name, prompt, guidance, steps, neg_prompt],
        outputs=outputs,
        fn=inference,
        cache_examples=False,
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
