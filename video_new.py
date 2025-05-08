import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import threading
import re
import cv2

# Load model and labels
model = tf.keras.models.load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

class DeepfakeDetector(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Traffic Classification")
        self.geometry("900x700")
        self.configure(bg="#2C2F33")

        # Initialize variables
        self.current_image = None
        self.current_video = None

        # UI
        self._create_widgets()
        self._setup_drag_drop()

    def _create_widgets(self):
        self.title_label = ctk.CTkLabel(self, text="Traffic Classification", font=("Arial", 22, "bold"), text_color="white")
        self.title_label.pack(pady=20)

        self.image_frame = ctk.CTkFrame(self, fg_color="#40444B", corner_radius=10)
        self.image_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.input_label = ctk.CTkLabel(self.image_frame, text="Drag & Drop or Click to Select an Image", font=("Arial", 16), text_color="white")
        self.input_label.pack(pady=10)

        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack()

        self.result_frame = ctk.CTkFrame(self, fg_color="#7289DA", corner_radius=10)
        self.result_frame.pack(pady=20)

        self.result_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 20), text_color="white")
        self.result_label.pack()

        self.confidence_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 18), text_color="white")

        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=10)

        self.upload_btn = ctk.CTkButton(self.button_frame, text="Select Image", command=self._browse_files, fg_color="#FF5733", text_color="white")
        self.upload_btn.pack(side="left", padx=10)

        self.detect_btn = ctk.CTkButton(self.button_frame, text="Detect", command=self._start_detection, fg_color="#FF5733", text_color="white")
        self.detect_btn.pack(side="left", padx=10)

        self.video_btn = ctk.CTkButton(self.button_frame, text="Select Video", command=self._browse_video, fg_color="#33C1FF", text_color="white")
        self.video_btn.pack(side="left", padx=10)

        self.video_detect_btn = ctk.CTkButton(self.button_frame, text="Detect Video", command=self._start_video_detection, fg_color="#33C1FF", text_color="white")
        self.video_detect_btn.pack(side="left", padx=10)

    def _setup_drag_drop(self):
        self.image_label.drop_target_register(DND_FILES)
        self.image_label.dnd_bind('<<Drop>>', self._handle_drop)
        self.image_label.bind('<Button-1>', lambda e: self._browse_files())

    def _handle_drop(self, event):
        file_path = re.sub(r'[{}]', '', event.data)
        self._load_image(file_path)

    def _browse_files(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self._load_image(file_path)

    def _browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if file_path:
            self.current_video = file_path
            self.input_label.configure(text=f"Selected video:\n{file_path.split('/')[-1]}")
            self.result_label.configure(text="")
            self.confidence_label.pack_forget()

    def _load_image(self, file_path):
        try:
            image = Image.open(file_path)
            self.current_image = image

            ctk_image = ctk.CTkImage(light_image=image, size=(400, 400))
            self.image_label.configure(image=ctk_image)
            self.image_label.image = ctk_image

            self.input_label.configure(text="")
            self.result_label.configure(text="")
            self.confidence_label.pack_forget()
        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")

    def _start_detection(self):
        if not self.current_image:
            self.show_error("Please upload an image first!")
            return

        self.detect_btn.configure(state="disabled")
        thread = threading.Thread(target=self._run_detection)
        thread.daemon = True
        thread.start()

    def _run_detection(self):
        try:
            size = (224, 224)
            image = ImageOps.fit(self.current_image, size, Image.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]

            self.after(0, self._show_result, index, confidence_score)
        except Exception as e:
            self.after(0, self.show_error, f"Error during detection: {str(e)}")
        finally:
            self.after(0, self._cleanup_detection)

    def _start_video_detection(self):
        if not self.current_video:
            self.show_error("Please upload a video first!")
            return

        self.video_detect_btn.configure(state="disabled")
        thread = threading.Thread(target=self._run_video_detection)
        thread.daemon = True
        thread.start()

    def _run_video_detection(self):
        try:
            cap = cv2.VideoCapture(self.current_video)
            if not cap.isOpened():
                raise Exception("Failed to open video")

            size = (224, 224)
            result_counts = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image = ImageOps.fit(image, size, Image.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                prediction = model.predict(data)
                index = np.argmax(prediction)
                result_counts[index] = result_counts.get(index, 0) + 1

            cap.release()

            final_index = max(result_counts, key=result_counts.get)
            confidence_score = result_counts[final_index] / sum(result_counts.values())

            self.after(0, self._show_result, final_index, confidence_score)
        except Exception as e:
            self.after(0, self.show_error, f"Video detection error: {str(e)}")
        finally:
            self.after(0, self._cleanup_video_detection)

    def _show_result(self, index, confidence_score):
        if index < len(class_names):
            class_name = class_names[index]
            self.result_label.configure(text=f"Predicted Class: {class_name}")

            if confidence_score > 0.90:
                self.confidence_label.configure(text=f"Confidence: {confidence_score:.4f}")
                self.confidence_label.pack()
            else:
                self.confidence_label.pack_forget()

    def _cleanup_detection(self):
        self.detect_btn.configure(state="normal")

    def _cleanup_video_detection(self):
        self.video_detect_btn.configure(state="normal")

    def show_error(self, message):
        self.result_label.configure(text=f"⚠️ {message}", text_color="red")


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = DeepfakeDetector()
    app.mainloop()
