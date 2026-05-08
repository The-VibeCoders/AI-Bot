import os
import time
import uuid
import torch
import gc
import random
from PIL import PngImagePlugin
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from app.core.config import SD_MODEL_ID, BASE_DIR

class ImageService:
    def __init__(self):
        self.sd_pipeline = None
        self.sd_model_loaded = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def unload(self):
        if self.sd_pipeline is not None:
            try: self.sd_pipeline.to("cpu")
            except Exception: pass
            self.sd_pipeline, self.sd_model_loaded = None, None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def draw(self, prompt: str, seed_override: int | None = None) -> tuple[str, str | None]:
        prompt = prompt.strip()
        if not prompt: return "[ERROR] Please provide a valid prompt.", None

        QUALITY = "masterpiece, best quality, ultra-detailed, photorealistic, 8k uhd, "
        NEG = "cgi, 3d, sketch, cartoon, anime, text, worst quality, low quality, ugly, duplicate, morbid, mutilated, poorly drawn, deformed, blurry, bad anatomy, extra limbs, disfigured"
        enriched = QUALITY + prompt

        try:
            if self.sd_model_loaded != SD_MODEL_ID or self.sd_pipeline is None:
                self.unload()
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=self.torch_dtype, safety_checker=None).to(self.device)
                self.sd_pipeline.enable_attention_slicing(slice_size="auto")
                self.sd_pipeline.enable_vae_tiling()
                self.sd_pipeline.enable_vae_slicing()
                if self.device == "cuda":
                    try: self.sd_pipeline.enable_xformers_memory_efficient_attention()
                    except: pass

                self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.sd_pipeline.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
                target = max(1, self.sd_pipeline.text_encoder.config.num_hidden_layers - 1)
                self.sd_pipeline.text_encoder.config.num_hidden_layers = target
                self.sd_model_loaded = SD_MODEL_ID

            seed = (int(seed_override) % (2 ** 32)) if seed_override is not None else random.randint(0, 2 ** 32 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            result = self.sd_pipeline(prompt=enriched, negative_prompt=NEG, num_inference_steps=20, guidance_scale=7.5, width=512, height=512, generator=generator)

            filename = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            file_path = os.path.join(BASE_DIR, filename)

            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt", enriched)
            meta.add_text("seed", str(seed))
            meta.add_text("model", SD_MODEL_ID)
            result.images[0].save(file_path, pnginfo=meta)

            return f"🎨 Saved '{filename}'\nSeed: {seed} | 512x512", filename
        except Exception as e:
            # If we get an out of memory error on CPU, we still unload and return the error.
            self.unload()
            return f"[ERROR] Image Generation Error: {e}", None
        
