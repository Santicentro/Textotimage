from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
image = pipeline(
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
).images[0]

image.save("generated_imagexl.png")