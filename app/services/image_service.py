import os
import time
import uuid
import torch
import gc
import random
from PIL import PngImagePlugin
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from app.core.config import SD_MODEL_ID, BASE_DIR

class ImageService:
    def __init__(self):
        self.sd_pipeline = None
        self.sd_img2img_pipeline = None
        self.sd_model_loaded = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def unload(self):
        if self.sd_pipeline is not None:
            try: self.sd_pipeline.to("cpu")
            except Exception: pass
            self.sd_pipeline = None
        if self.sd_img2img_pipeline is not None:
            try: self.sd_img2img_pipeline.to("cpu")
            except Exception: pass
            self.sd_img2img_pipeline = None
        self.sd_model_loaded = None
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
                self.sd_pipeline.vae.enable_tiling()
                self.sd_pipeline.vae.enable_slicing()
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

    def edit_image(self, image_path: str, prompt: str, strength: float = 0.55, seed_override: int | None = None, high_quality: bool = True) -> tuple[str, str | None]:
        """Edit an existing image using Stable Diffusion with your text prompt.

        Args:
            image_path: Path to the uploaded image
            prompt: Your editing prompt describing what to change (e.g., "person drinking coffee")
            strength: How much to transform (0.1=subtle, 0.8=drastic). Default 0.55 for consistency
            seed_override: Optional seed for reproducibility
            high_quality: Use higher resolution and steps for better quality
        """
        prompt = prompt.strip()
        if not prompt: return "[ERROR] Please provide a valid prompt.", None

        if not os.path.exists(image_path): return "[ERROR] Image file not found.", None

        try:
            from PIL import Image as PILImage, ImageEnhance
            init_image = PILImage.open(image_path).convert("RGB")

            # Resize for optimal processing while maintaining aspect ratio
            max_size = 1024 if self.device == "cuda" else 768
            if max(init_image.size) > max_size:
                init_image.thumbnail((max_size, max_size), PILImage.LANCZOS)

            # Ensure dimensions are multiples of 8 (SD requirement)
            w, h = init_image.size
            new_w, new_h = (w // 8) * 8, (h // 8) * 8
            if new_w != w or new_h != h:
                init_image = init_image.resize((new_w, new_h), PILImage.LANCZOS)

            # Pre-process for better consistency
            enhancer = ImageEnhance.Contrast(init_image)
            init_image = enhancer.enhance(1.05)

            # Build comprehensive prompt for better context understanding
            QUALITY = "masterpiece, best quality, ultra-detailed, photorealistic, sharp focus, "
            STYLE = "professional photography, natural lighting, "
            SUBJECT = ""

            # Detect what kind of edit the user wants and frame it properly
            # Lower case for detection
            prompt_lower = prompt.lower()

            # If user mentions a person/portrait scenario
            if any(word in prompt_lower for word in ['person', 'man', 'woman', 'girl', 'boy', 'human', 'face', 'portrait', ' selfie', 'drinking', 'eating', 'holding', 'wearing', 'sitting', 'standing', 'walking', 'smiling', 'looking', 'staring']):
                SUBJECT = "person, "
            # If it's a scene/environment edit
            elif any(word in prompt_lower for word in ['background', 'sky', 'scene', 'landscape', 'room', 'building', 'outdoor', 'indoor', 'beach', 'city', 'forest']):
                SUBJECT = "scene, "
            # If it's an object edit
            elif any(word in prompt_lower for word in ['car', 'dog', 'cat', 'object', 'flower', 'tree', 'car']):
                SUBJECT = "object, "

            # Add action context based on prompt keywords
            ACTION = ""
            if 'drinking' in prompt_lower or 'coffee' in prompt_lower:
                ACTION = "holding a coffee cup, taking a sip of coffee, "
            if 'eating' in prompt_lower or 'food' in prompt_lower:
                ACTION += "eating food, "
            if 'smiling' in prompt_lower:
                ACTION += "smiling, happy expression, "
            if 'wearing' in prompt_lower:
                ACTION += "wearing "
                for word in prompt.split():
                    if word not in ['wearing', 'is', 'a', 'the', 'and', 'with']:
                        ACTION += word + " "
                ACTION += ", "

            # Build the enriched prompt
            enriched = QUALITY + STYLE + SUBJECT + ACTION + prompt

            # Enhanced negative prompts for consistency
            NEG = "cgi, 3d, cartoon, anime, sketch, painting, illustration, art, "
            NEG += "text, watermark, logo, signature, cropped, out of frame, "
            NEG += "worst quality, low quality, blurry, bad anatomy, deformed, "
            NEG += "extra limbs, disfigured, poorly drawn face, distorted face, "
            NEG += "noise, artifacts, jpeg artifacts, oversaturated, washed out"

            if self.sd_img2img_pipeline is None or self.sd_model_loaded != SD_MODEL_ID:
                self.unload()
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    SD_MODEL_ID, torch_dtype=self.torch_dtype, safety_checker=None
                ).to(self.device)
                self.sd_pipeline.enable_attention_slicing(slice_size="auto")
                self.sd_pipeline.vae.enable_tiling()
                self.sd_pipeline.vae.enable_slicing()
                if self.device == "cuda":
                    try: self.sd_pipeline.enable_xformers_memory_efficient_attention()
                    except: pass

                self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.sd_pipeline.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
                )
                target = max(1, self.sd_pipeline.text_encoder.config.num_hidden_layers - 1)
                self.sd_pipeline.text_encoder.config.num_hidden_layers = target
                self.sd_img2img_pipeline = StableDiffusionImg2ImgPipeline(
                    vae=self.sd_pipeline.vae,
                    text_encoder=self.sd_pipeline.text_encoder,
                    tokenizer=self.sd_pipeline.tokenizer,
                    unet=self.sd_pipeline.unet,
                    scheduler=self.sd_pipeline.scheduler,
                    safety_checker=None,
                    feature_extractor=self.sd_pipeline.feature_extractor,
                ).to(self.device)
                self.sd_model_loaded = SD_MODEL_ID

            seed = (int(seed_override) % (2 ** 32)) if seed_override is not None else random.randint(0, 2 ** 32 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Optimized settings for subject/action edits
            num_steps = 35 if high_quality else 25
            guidance_scale = 7.5  # Higher CFG for better prompt adherence on actions

            result = self.sd_img2img_pipeline(
                prompt=enriched,
                negative_prompt=NEG,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator
            )

            output_image = result.images[0]

            # Post-process
            enhancer = ImageEnhance.Sharpness(output_image)
            output_image = enhancer.enhance(1.1)

            filename = f"edit_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            file_path = os.path.join(BASE_DIR, filename)

            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt", enriched)
            meta.add_text("user_prompt", prompt)
            meta.add_text("seed", str(seed))
            meta.add_text("model", SD_MODEL_ID)
            meta.add_text("original", image_path)
            meta.add_text("strength", str(strength))
            meta.add_text("steps", str(num_steps))
            output_image.save(file_path, pnginfo=meta, optimize=True)

            return f"✏️ Edited '{filename}'\nPrompt: {enriched}\nSeed: {seed} | {output_image.size[0]}x{output_image.size[1]}", filename
        except Exception as e:
            self.unload()
            return f"[ERROR] Image Edit Error: {e}", None
        
