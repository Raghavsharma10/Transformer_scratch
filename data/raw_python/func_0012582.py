def save_image(self, cat, img, data):
        """Saves a new image."""
        filename = self.path(cat, img)
        mkdir(filename)
        if type(data) == np.ndarray:
            data = Image.fromarray(data).convert('RGB')
        data.save(filename)