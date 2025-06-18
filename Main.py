import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import string
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_ui()
        self.create_particles()
        self.model = None
        self.tfidf = None
        self.model_dir = "spam_model"
        self.model_path = os.path.join(self.model_dir, "model.pkl")
        self.tfidf_path = os.path.join(self.model_dir, "tfidf.pkl")
        self._load_pretrained_model()
        self.animate_particles()

    def setup_window(self):
        """Configure the main window"""
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Window setup
        self.root.title("Spam Detector")
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.root.resizable(True, True)
        self.root.configure(bg="black")

    def setup_ui(self):
        """Create all UI elements"""
        # Canvas for background effects
        self.canvas = tk.Canvas(self.root, bg="black", width=self.screen_width,
                                 height=self.screen_height, highlightthickness=0)
        self.canvas.place(x=0, y=0)

        # Large "SPAM DETECTOR" faded background title
        self.truth_label = tk.Label(self.root, text="SPAM DETECTOR",
                                     font=("Helvetica", int(self.screen_width / 12), "bold"),
                                     fg="#444444", bg="black")
        self.truth_label.place(relx=0.5, rely=0.25, anchor="center")

        # Input frame
        input_frame = tk.Frame(self.root, bg="black")
        input_frame.place(relx=0.5, rely=0.5, anchor="center", width=int(self.screen_width * 0.8))

        # Message label
        message_label = tk.Label(input_frame, text="Enter a message to check:",
                                     font=("Arial", 14), fg="white", bg="black")
        message_label.pack(pady=(0, 5), anchor="w")

        # Input field
        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(input_frame, textvariable=self.entry_var,
                                 font=("Arial", 22), width=60)
        self.entry.pack(fill="x", pady=5)
        self.entry.bind("<Return>", lambda event: self.check_message())

        # Button style
        style = ttk.Style()
        style.configure("Modern.TButton",
                        font=("Arial", 18, "bold"),
                        padding=12,
                        foreground="black",
                        background="white",
                        borderwidth=0)
        style.map("Modern.TButton",
                  background=[("active", "#e0e0e0")],
                  foreground=[("disabled", "gray")])

        # Button frame
        button_frame = tk.Frame(input_frame, bg="black")
        button_frame.pack(pady=15)

        # Check button
        self.check_button = ttk.Button(button_frame, text="CHECK",
                                         style="Modern.TButton",
                                         command=self.check_message,
                                         state="disabled")  # Initially disabled
        self.check_button.pack(side="left", padx=10)

        # Result label
        self.result_label = tk.Label(self.root, text="",
                                      font=("Helvetica", 24, "bold"), bg="black")
        self.result_label.place(relx=0.5, rely=0.7, anchor="center")

        # Model status label
        self.status_label = tk.Label(self.root, text="No model loaded",
                                       font=("Arial", 12), fg="white", bg="black")
        self.status_label.place(relx=0.5, rely=0.8, anchor="center")

    def create_particles(self):
        """Create particle effect for background"""
        self.particles = []
        num_particles = 120

        for _ in range(num_particles):
            x = random.randint(-100, 150) if random.random() < 0.5 else random.randint(self.screen_width - 150, self.screen_width + 100)
            y = random.randint(-50, self.screen_height + 50)
            size = random.randint(2, 6)
            speed = random.uniform(0.8, 2.5)
            p = self.canvas.create_oval(x, y, x + size, y + size, fill="white", outline="")
            self.particles.append((p, speed))

    def animate_particles(self):
        """Move particles from top to bottom. Reset them to top once they exit the screen."""
        for p, speed in self.particles:
            self.canvas.move(p, 0, speed)
            try:
                x, y, x1, y1 = self.canvas.coords(p)
                if y > self.screen_height:
                    new_y = random.randint(-100, -50)
                    self.canvas.coords(p, x, new_y, x1, new_y + (y1 - y))
            except:
                pass  # Handle any issues with particle coordinates
        self.root.after(30, self.animate_particles)

    def _load_pretrained_model(self):
        """Load the pre-trained model and TF-IDF vectorizer"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.tfidf_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)

                with open(self.tfidf_path, 'rb') as f:
                    self.tfidf = pickle.load(f)

                self.check_button.config(state="normal")
                self.status_label.config(text="Pre-trained model loaded successfully!")
                return True
            else:
                self.status_label.config(text="Pre-trained model files not found.")
                return False
        except Exception as e:
            self.status_label.config(text=f"Could not load pre-trained model: {str(e)}")
            return False

    def check_message(self):
        """Check if input message is spam or ham"""
        if not self.model or not self.tfidf:
            messagebox.showwarning("Warning", "Model not loaded. Ensure model.pkl and tfidf.pkl are in the spam_model directory.")
            return

        text = self.entry_var.get()
        if not text:
            messagebox.showinfo("Info", "Please enter a message to check.")
            return

        try:
            # Preprocess input
            text = text.lower()
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)

            # Vectorize and predict
            input_vector = self.tfidf.transform([text])
            prediction = self.model.predict(input_vector)[0]
            probability = self.model.predict_proba(input_vector)[0]

            # Update result label
            if prediction == 1:
                spam_prob = probability[1] * 100
                self.result_label.config(text=f"SPAM ({spam_prob:.1f}%)", fg="red")
            else:
                ham_prob = probability[0] * 100
                self.result_label.config(text=f"HAM ({ham_prob:.1f}%)", fg="lightgreen")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")

def main():
    # Create root window
    root = tk.Tk()

    # Set window title
    root.title("Spam Detector")

    app = SpamDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
