def data(self, value):
        """
        Saves a new image to disk
        """
        self.loader.save_image(self.category, self.image, value)