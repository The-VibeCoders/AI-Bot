import os
import uuid
import tempfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
from typing import Optional, Tuple

class ImageEditor:
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "image_edits")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_output_path(self, filename: str) -> str:
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        return os.path.join(self.output_dir, unique_name)

    def _validate_image(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in self.SUPPORTED_FORMATS

    def load_image(self, filepath: str) -> Optional[Image.Image]:
        if not self._validate_image(filepath):
            return None
        try:
            return Image.open(filepath)
        except Exception:
            return None

    def save_image(self, image: Image.Image, original_filename: str) -> str:
        output_path = self._get_output_path(original_filename)
        image.save(output_path)
        return output_path

    def resize(self, filepath: str, width: int, height: int, maintain_aspect: bool = True) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        if maintain_aspect:
            img.thumbnail((width, height), Image.LANCZOS)
        else:
            img = img.resize((width, height), Image.LANCZOS)
        return self.save_image(img, os.path.basename(filepath))

    def crop(self, filepath: str, x: int, y: int, width: int, height: int) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        cropped = img.crop((x, y, x + width, y + height))
        return self.save_image(cropped, os.path.basename(filepath))

    def rotate(self, filepath: str, degrees: float) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        rotated = img.rotate(degrees, expand=True)
        return self.save_image(rotated, os.path.basename(filepath))

    def flip_horizontal(self, filepath: str) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.save_image(flipped, os.path.basename(filepath))

    def flip_vertical(self, filepath: str) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        return self.save_image(flipped, os.path.basename(filepath))

    def blur(self, filepath: str, radius: int = 2) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return self.save_image(blurred, os.path.basename(filepath))

    def sharpen(self, filepath: str, factor: float = 1.5) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        enhancer = ImageEnhance.Sharpness(img)
        sharpened = enhancer.enhance(factor)
        return self.save_image(sharpened, os.path.basename(filepath))

    def adjust_brightness(self, filepath: str, factor: float = 1.2) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        enhancer = ImageEnhance.Brightness(img)
        adjusted = enhancer.enhance(factor)
        return self.save_image(adjusted, os.path.basename(filepath))

    def adjust_contrast(self, filepath: str, factor: float = 1.2) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        enhancer = ImageEnhance.Contrast(img)
        adjusted = enhancer.enhance(factor)
        return self.save_image(adjusted, os.path.basename(filepath))

    def grayscale(self, filepath: str) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        gray = ImageOps.grayscale(img)
        gray_img = gray.convert("RGB")
        return self.save_image(gray_img, os.path.basename(filepath))

    def add_text(self, filepath: str, text: str, x: int = 10, y: int = 10,
                 font_size: int = 24, color: Tuple[int, int, int] = (255, 255, 255)) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x, y), text, fill=color, font=font)
        return self.save_image(img, os.path.basename(filepath))

    def add_border(self, filepath: str, border_width: int = 5,
                   border_color: Tuple[int, int, int] = (0, 0, 0)) -> Optional[str]:
        img = self.load_image(filepath)
        if not img:
            return None
        bordered = ImageOps.expand(img, border=border_width, fill=border_color)
        return self.save_image(bordered, os.path.basename(filepath))


# Global instance
image_editor = ImageEditor()