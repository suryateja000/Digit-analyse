fimport tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

class DigitDrawer:
    def __init__(self):
        try:
            self.model = tf.keras.models.load_model('digit_recognition.keras')
            print("Model loaded successfully!")
        except:
            print("Error: Model file not found!")
            return
            
        self.root = tk.Tk()
        self.root.title("Digit Recognition")
        self.root.geometry("350x500")
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        self.image = Image.new("L", (280, 280), 255)  
        self.draw_pil = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        tk.Button(self.root, text="Predict", command=self.predict, 
                 bg='green', fg='white', font=('Arial', 12)).pack(pady=10)
        tk.Button(self.root, text="Clear", command=self.clear, 
                 bg='red', fg='white', font=('Arial', 12)).pack(pady=5)
        self.result = tk.Label(self.root, text="Draw a digit and click Predict", 
                              font=('Arial', 14), bg='lightgray', pady=10)
        self.result.pack(pady=20, fill='x', padx=20)
        self.debug_canvas = tk.Canvas(self.root, width=140, height=140, bg='black')
        self.debug_canvas.pack(pady=10)
        tk.Label(self.root, text="Processed Image (28x28)", font=('Arial', 10)).pack()
    
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black', outline='black')
        self.draw_pil.ellipse([x-8, y-8, x+8, y+8], fill=0)  
    
    def clear(self):
        self.canvas.delete("all")
        self.debug_canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  
        self.draw_pil = ImageDraw.Draw(self.image)
        self.result.config(text="Draw a digit and click Predict")
    
    def predict(self):
        try:
            img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)
            
            img_array = 255 - img_array
            
            img_array = img_array / 255.0
            
            img_array = img_array.reshape(1, 28, 28)
            
            self.show_processed_image(img_array[0])
            
            prediction = self.model.predict(img_array, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            self.result.config(text=f"Predicted: {digit} (Confidence: {confidence:.1f}%)", 
                              fg='blue', font=('Arial', 14, 'bold'))
            
        except Exception as e:
            self.result.config(text=f"Error: {str(e)}", fg='red')
            print(f"Error: {e}")
    
    def show_processed_image(self, img_28x28):
        """Show how the image looks after processing"""
        self.debug_canvas.delete("all")
        
        img_scaled = np.repeat(np.repeat(img_28x28, 5, axis=0), 5, axis=1)
        
        for i in range(140):
            for j in range(140):
                intensity = int(img_scaled[i//5, j//5] * 255)
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                self.debug_canvas.create_rectangle(j, i, j+1, i+1, fill=color, outline=color)
    
    def run(self):
        self.root.mainloop()

app = DigitDrawer()
app.run()
