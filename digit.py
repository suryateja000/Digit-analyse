import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

class ProDigitRecognizer:
    COLOR_BG = '#FFFFFF'
    COLOR_PANE_BG = '#F7F7F7'
    COLOR_CANVAS_BG = '#121212'
    COLOR_ACCENT = '#007AFF'
    COLOR_TEXT = '#000000'
    COLOR_TEXT_MUTED = '#6d6d6d'
    COLOR_BORDER = '#EAEAEA'
    FONT_FAMILY = 'Segoe UI'

    def __init__(self):
        try:
            self.model = tf.keras.models.load_model('digit_recognition.keras')
        except IOError:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Fatal Error", "'digit_recognition.keras' not found.")
            root.destroy()
            return
            
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")
        self.root.geometry("800x600")
        self.root.configure(bg=self.COLOR_BG)
        self.root.minsize(800, 600)

        self.font_header = tkfont.Font(family=self.FONT_FAMILY, size=20, weight='bold')
        self.font_subheader = tkfont.Font(family=self.FONT_FAMILY, size=11, weight='normal')
        self.font_result = tkfont.Font(family=self.FONT_FAMILY, size=120, weight='bold')
        self.font_confidence = tkfont.Font(family=self.FONT_FAMILY, size=14, weight='bold')
        self.font_button = tkfont.Font(family=self.FONT_FAMILY, size=11, weight='bold')

        self._create_layout()

    def _create_layout(self):
        self.root.grid_columnconfigure(0, weight=1, uniform='group1')
        self.root.grid_columnconfigure(1, weight=1, uniform='group1')
        self.root.grid_rowconfigure(0, weight=1)

        self._create_canvas_pane()
        self._create_control_pane()

    def _create_canvas_pane(self):
        canvas_pane = tk.Frame(self.root, bg=self.COLOR_BG, padx=40, pady=40)
        canvas_pane.grid(row=0, column=0, sticky='nsew')
        canvas_pane.grid_rowconfigure(0, weight=1)
        canvas_pane.grid_columnconfigure(0, weight=1)

        canvas_container = tk.Frame(canvas_pane, bg=self.COLOR_CANVAS_BG, relief='flat')
        canvas_container.grid(row=0, column=0, sticky='nsew')
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_container, bg=self.COLOR_CANVAS_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        self.image = Image.new("L", (512, 512), "black") 
        self.draw_pil = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.canvas.config(cursor="arrow"))

    def _create_control_pane(self):
        control_pane = tk.Frame(self.root, bg=self.COLOR_PANE_BG, padx=40, pady=40)
        control_pane.grid(row=0, column=1, sticky='nsew')
        control_pane.grid_columnconfigure(0, weight=1)
        
        tk.Label(control_pane, text="AI RECOGNITION", font=self.font_header, 
                 fg=self.COLOR_TEXT, bg=self.COLOR_PANE_BG).grid(row=0, column=0, sticky='w')
        
        tk.Label(control_pane, text="Draw a single digit on the black canvas.", font=self.font_subheader, 
                 fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_PANE_BG).grid(row=1, column=0, sticky='w', pady=(0, 40))

        self.result_frame = tk.Frame(control_pane, bg=self.COLOR_PANE_BG)
        self.result_frame.grid(row=2, column=0, sticky='nsew', pady=40)
        self.result_frame.grid_columnconfigure(0, weight=1)
        self._show_initial_result_state()

        button_frame = tk.Frame(control_pane, bg=self.COLOR_PANE_BG)
        button_frame.grid(row=3, column=0, sticky='sew')
        button_frame.grid_columnconfigure((0,1), weight=1)
        
        StyledButton(button_frame, text="Clear Canvas", command=self._clear, 
                     bg_color=self.COLOR_BG, fg_color=self.COLOR_TEXT, border_color=self.COLOR_BORDER, font=self.font_button
        ).grid(row=0, column=0, sticky='ew', padx=(0, 10))
        
        StyledButton(button_frame, text="Predict Digit", command=self._predict, 
                     bg_color=self.COLOR_ACCENT, fg_color='#FFFFFF', border_color=self.COLOR_ACCENT, font=self.font_button
        ).grid(row=0, column=1, sticky='ew', padx=(10, 0))

    def _show_initial_result_state(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        tk.Label(self.result_frame, text="?", font=self.font_result, 
                 fg=self.COLOR_BORDER, bg=self.COLOR_PANE_BG).pack()
        tk.Label(self.result_frame, text="Awaiting Input", font=self.font_confidence, 
                 fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_PANE_BG).pack()

    def _draw(self, event):
        self.canvas.config(cursor="crosshair")
        x, y = event.x, event.y
        r = int(self.canvas.winfo_width() * 0.04) 
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        
        self.draw_pil.ellipse([x-r, y-r, x+r, y+r], fill='white')

    def _clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (512, 512), "black")
        self.draw_pil = ImageDraw.Draw(self.image)
        self._show_initial_result_state()

    def _predict(self):
        bbox = self.image.getbbox()
        if not bbox: return

        img_cropped = self.image.crop(bbox)
        
        width, height = img_cropped.size
        size = max(width, height) + 40 
        img_padded = Image.new("L", (size, size), "black")
        img_padded.paste(img_cropped, ((size - width) // 2, (size - height) // 2))

        img_resized = img_padded.resize((28, 28), Image.Resampling.LANCZOS)

        img_array = np.array(img_resized) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = self.model.predict(img_array, verbose=0)[0]
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        self._update_result_display(digit, confidence)

    def _update_result_display(self, digit, confidence):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        tk.Label(self.result_frame, text=f"{digit}", font=self.font_result, 
                 fg=self.COLOR_TEXT, bg=self.COLOR_PANE_BG).pack()
        tk.Label(self.result_frame, text=f"{confidence:.1%} Confidence", font=self.font_confidence, 
                 fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_PANE_BG).pack()

    def run(self):
        if hasattr(self, 'root'):
            self.root.mainloop()

class StyledButton(tk.Frame):
    """A professional, modern button with hover effects."""
    def __init__(self, parent, text, command, bg_color, fg_color, border_color, font):
        super().__init__(parent, bg=border_color)
        
        self.original_bg = bg_color
        self.hover_bg = self._adjust_brightness(bg_color, 0.9 if bg_color != ProDigitRecognizer.COLOR_BG else 0.95)
        
        self.inner_frame = tk.Frame(self, bg=self.original_bg)
        self.inner_frame.pack(fill='both', expand=True, padx=1, pady=1)
        
        self.label = tk.Label(self.inner_frame, text=text, font=font,
                              bg=self.original_bg, fg=fg_color, pady=15, cursor='hand2')
        self.label.pack(fill='both', expand=True)

        self.inner_frame.bind("<Enter>", self._on_hover)
        self.inner_frame.bind("<Leave>", self._on_leave)
        self.inner_frame.bind("<Button-1>", lambda e: command())
        self.label.bind("<Button-1>", lambda e: command())

    def _on_hover(self, event):
        self.inner_frame.config(bg=self.hover_bg)
        self.label.config(bg=self.hover_bg)

    def _on_leave(self, event):
        self.inner_frame.config(bg=self.original_bg)
        self.label.config(bg=self.original_bg)

    def _adjust_brightness(self, hex_color, factor):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r, g, b = [max(0, min(255, int(c * factor))) for c in (r, g, b)]
        return f"#{r:02x}{g:02x}{b:02x}"

if __name__ == "__main__":
    app = ProDigitRecognizer()
    app.run()
