import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io

app = FastAPI()


@app.post("/generate")
async def generate_image(prompt: str = Form(...), sketch: UploadFile = File(...)):
    sketch_bytes = await sketch.read()
    sketch_img = (
        Image.open(io.BytesIO(sketch_bytes)).convert("RGB").resize((1024, 1024))
    )

    generated = await generate(prompt=prompt, image=sketch_img)
    buf = io.BytesIO()
    generated.save(buf, format="PNG")
    return StreamingResponse(buf, media_type="image/png")


async def generate(image, prompt):
    from diffusers import (
        StableDiffusionXLAdapterPipeline,
        T2IAdapter,
        EulerAncestralDiscreteScheduler,
        AutoencoderKL,
    )
    from diffusers.utils import load_image, make_image_grid
    from controlnet_aux.pidi import PidiNetDetector
    import torch

    # load adapter

    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        torch_dtype=torch.float16,
        varient="fp16",
    ).to("cuda")
    # load euler_a scheduler
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id,
        vae=vae,
        adapter=adapter,
        scheduler=euler_a,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
    url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_sketch.png"
    # image = load_image(url)
    image = pidinet(
        image, detect_resolution=1024, image_resolution=1024, apply_filter=True
    )

    # prompt = "a robot, mount fuji in the background, 4k photo, highly detailed"
    negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"
    gen_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=30,
        adapter_conditioning_scale=0.9,
        guidance_scale=7.5,
    ).images[0]

    gen_images.save("out_sketch.png")
    return gen_images


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
