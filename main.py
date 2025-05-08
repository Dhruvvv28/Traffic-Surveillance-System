import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import threading
import re

# Load the model
model = tf.keras.models.load_model("keras_Model.h5", compile=False)

# Load labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

class DeepfakeDetector(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Traffic Classification ")
        self.geometry("900x700")
        self.configure(bg="#2C2F33")  # Dark Background

        # Initialize variables
        self.current_image = None

        # UI Elements
        self._create_widgets()
        self._setup_drag_drop()

    def _create_widgets(self):
        """Create and arrange all widgets"""
        # Title
        self.title_label = ctk.CTkLabel(
            self, text="Traffic Classification",
            font=("Arial", 22, "bold"),
            text_color="white"
        )
        self.title_label.pack(pady=20)

        # Image Preview Frame
        self.image_frame = ctk.CTkFrame(self, fg_color="#40444B", corner_radius=10)
        self.image_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Image Label
        self.input_label = ctk.CTkLabel(
            self.image_frame, text="Drag & Drop or Click to Select an Image",
            font=("Arial", 16), text_color="white"
        )
        self.input_label.pack(pady=10)

        # Image Display
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack()

        # Results Frame
        self.result_frame = ctk.CTkFrame(self, fg_color="#7289DA", corner_radius=10)
        self.result_frame.pack(pady=20)

        # Predicted Class Label
        self.result_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 20), text_color="white")
        self.result_label.pack()

        # Confidence Score Label
        self.confidence_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 18), text_color="white")

        # Buttons
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=10)

        self.upload_btn = ctk.CTkButton(
            self.button_frame, text="Select Image",
            command=self._browse_files, fg_color="#FF5733", text_color="white"
        )
        self.upload_btn.pack(side="left", padx=10)

        self.detect_btn = ctk.CTkButton(
            self.button_frame, text="Detect",
            command=self._start_detection, fg_color="#FF5733", text_color="white"
        )
        self.detect_btn.pack(side="left", padx=10)

    def _setup_drag_drop(self):
        """Setup drag and drop functionality"""
        self.image_label.drop_target_register(DND_FILES)
        self.image_label.dnd_bind('<<Drop>>', self._handle_drop)
        self.image_label.bind('<Button-1>', lambda e: self._browse_files())

    def _handle_drop(self, event):
        """Handle dropped files"""
        file_path = re.sub(r'[{}]', '', event.data)
        self._load_image(file_path)

    def _browse_files(self):
        """Open file browser"""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self._load_image(file_path)

    def _load_image(self, file_path):
        """Load and display image"""
        try:
            image = Image.open(file_path)
            self.current_image = image

            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=image, size=(400, 400))
            self.image_label.configure(image=ctk_image)
            self.image_label.image = ctk_image

            self.input_label.configure(text="")  # Hide text
            self.result_label.configure(text="")
            self.confidence_label.pack_forget()  # Hide confidence label
        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")

    def _start_detection(self):
        """Start deepfake detection in a separate thread"""
        if not self.current_image:
            self.show_error("Please upload an image first!")
            return

        self.detect_btn.configure(state="disabled")
        thread = threading.Thread(target=self._run_detection)
        thread.daemon = True
        thread.start()

    def _run_detection(self):
        """Run the deepfake detection"""
        try:
            # Process image
            size = (224, 224)
            image = ImageOps.fit(self.current_image, size, Image.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Prepare data for prediction
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predict
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]

            # Show result
            self.after(0, self._show_result, index, confidence_score)
        except Exception as e:
            self.after(0, self.show_error, f"Error during detection: {str(e)}")
        finally:
            self.after(0, self._cleanup_detection)

    def _show_result(self, index, confidence_score):
        """Display detection result"""
        if index < len(class_names):
            class_name = class_names[index]
            self.result_label.configure(text=f"Predicted Class: {class_name}")

            # Show confidence score only if above 90%
            if confidence_score > 0.90:
                self.confidence_label.configure(text=f"Confidence: {confidence_score:.4f}")
                self.confidence_label.pack()
            else:
                self.confidence_label.pack_forget()

    def _cleanup_detection(self):
        """Clean up after detection"""
        self.detect_btn.configure(state="normal")

    def show_error(self, message):
        """Display error message"""
        self.result_label.configure(text=f"⚠️ {message}", text_color="red")


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = DeepfakeDetector()
    app.mainloop()
