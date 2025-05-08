import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

def load_dataset():
    global train_generator, validation_generator, class_labels
    train_dir = filedialog.askdirectory(title='Select Training Data Folder')
    validation_dir = filedialog.askdirectory(title='Select Validation Data Folder')
    
    if not train_dir or not validation_dir:
        messagebox.showerror("Error", "Please select valid dataset folders")
        return
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    
    class_labels = sorted(train_generator.class_indices.keys())
    with open('lb.txt', 'w') as f:
        for label in class_labels:
            f.write(label + '\n')
    
    messagebox.showinfo("Success", "Dataset Loaded Successfully")

def train_model():
    global model
    if not class_labels:
        messagebox.showerror("Error", "Please load the dataset first")
        return
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(len(class_labels), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size)
    
    messagebox.showinfo("Training Complete", "Model trained successfully")

def evaluate_model():
    if 'model' not in globals():
        messagebox.showerror("Error", "Please train the model first")
        return
    
    test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
    messagebox.showinfo("Evaluation Result", f"Test Accuracy: {test_acc:.2f}")

def save_model():
    if 'model' not in globals():
        messagebox.showerror("Error", "Please train the model first")
        return
    model.save('image_classification_model.h5')
    messagebox.showinfo("Saved", "Model saved as image_classification_model.h5")

# GUI Setup
root = tk.Tk()
root.title("Training  ")
root.geometry("400x400")
root.configure(bg="#f0f8ff")

style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=5)

title_label = tk.Label(root, text="Training ", font=("Arial", 14, "bold"), bg="#f0f8ff")
title_label.pack(pady=10)

load_btn = ttk.Button(root, text="Load Dataset", command=load_dataset)
load_btn.pack(pady=5)

train_btn = ttk.Button(root, text="Train Model", command=train_model)
train_btn.pack(pady=5)

evaluate_btn = ttk.Button(root, text="Evaluate Model", command=evaluate_model)
evaluate_btn.pack(pady=5)

save_btn = ttk.Button(root, text="Save Model", command=save_model)
save_btn.pack(pady=5)

exit_btn = ttk.Button(root, text="Exit", command=root.quit)
exit_btn.pack(pady=10)

root.mainloop()
