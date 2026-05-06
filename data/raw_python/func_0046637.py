def load_image(canvas, filepath, bounds=None):
    """Takes a tk.Canvas and a filepath, loads image into canvas"""

    image_data = Image.open(filepath)
    if bounds:
        image_data.thumbnail(bounds, PIL.Image.ANTIALIAS)
    canvas.image = ImageTk.PhotoImage(image_data)
    canvas.create_image(0, 0, image=canvas.image, anchor=tk.NW)