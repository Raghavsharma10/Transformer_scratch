def load_image(self, imagepath, width=None, height=None):
        """Loads new image into canvas, updating size if needed."""

        if width:
            self.width = width
            self.canvas["width"] = width
        if height:
            self.height = height
            self.canvas["height"] = height

        self.image = imagepath
        size = (self.width, self.height)
        load_image(self.canvas, self.image, bounds=size)
        self.canvas.update_idletasks()