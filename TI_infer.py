from diffusers import StableDiffusionPipeline
import torch
import os

# model_id = "/share2/huangrenyuan/model_zoo/stable-diffusion-2-1-base"
model_id = "/share2/huangrenyuan/model_zoo/bk-sdm-v2-med"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda:0")

prompt = "a photo of a <painting>"
out_path = "samples/distillation/starry_night/sd_model_med_token"
learned_token_path = "invert_tokens/starry_night_sd2_1"
os.makedirs(out_path, exist_ok=True)
sample_prefix = "bk_sd"

# Load distilled weights
state_dict = torch.load(f"distillation/starry_night_test/checkpoint-2000/unet_ckpt.pt")
pipe.unet.load_state_dict(state_dict)

token_ckpt = "learned_embeds-steps-500.safetensors"
repo_id_embeds = os.path.join(learned_token_path, token_ckpt)
pipe.load_textual_inversion(repo_id_embeds)
generator = torch.Generator(device=pipe.device).manual_seed(3467)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
image.save(f"{out_path}/{sample_prefix}_500.png")
pipe.unload_textual_inversion()

token_ckpt = "learned_embeds-steps-1000.safetensors"
repo_id_embeds = os.path.join(learned_token_path, token_ckpt)
pipe.load_textual_inversion(repo_id_embeds)
generator = torch.Generator(device=pipe.device).manual_seed(3467)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
image.save(f"{out_path}/{sample_prefix}_1000.png")
pipe.unload_textual_inversion()

token_ckpt = "learned_embeds-steps-1500.safetensors"
repo_id_embeds = os.path.join(learned_token_path, token_ckpt)
pipe.load_textual_inversion(repo_id_embeds)
generator = torch.Generator(device=pipe.device).manual_seed(3467)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
image.save(f"{out_path}/{sample_prefix}_1500.png")
pipe.unload_textual_inversion()

token_ckpt = "learned_embeds-steps-2000.safetensors"
repo_id_embeds = os.path.join(learned_token_path, token_ckpt)
pipe.load_textual_inversion(repo_id_embeds)
generator = torch.Generator(device=pipe.device).manual_seed(3467)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
image.save(f"{out_path}/{sample_prefix}_2000.png")