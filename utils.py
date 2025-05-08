import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from typing import Tuple, Optional
import io

def create_loading_animation(parent, size: int = 30) -> ctk.CTkCanvas:
    """Create a circular loading animation"""
    canvas = ctk.CTkCanvas(parent, width=size, height=size, 
                          bg=parent.cget('fg_color'), highlightthickness=0)
    
    def update_loading(angle=0):
        canvas.delete("load")
        canvas.create_arc(4, 4, size-4, size-4, 
                         start=angle, extent=30,
                         tags="load", width=3,
                         style="arc")
        canvas.after(50, update_loading, (angle + 30) % 360)
    
    canvas.after(50, update_loading)
    return canvas

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model input"""
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def resize_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    """Resize image maintaining aspect ratio"""
    ratio = min(max_size[0]/image.width, max_size[1]/image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

class ImagePreview(ctk.CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(expand=True, fill="both", padx=10, pady=10)
        self.current_image = None
        
    def set_image(self, image: Image.Image):
        """Set and display the image"""
        self.current_image = image
        preview = resize_image(image, (300, 300))
        photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
