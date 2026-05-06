def display(self, image):
        """
        Takes a :py:mod:`PIL.Image` and dumps it to a numbered PNG file.
        """
        assert(image.size == self.size)
        self._last_image = image

        self._count += 1
        filename = self._file_template.format(self._count)
        image = self.preprocess(image)
        surface = self.to_surface(image, alpha=self._contrast)
        logger.debug("Writing: {0}".format(filename))
        self._pygame.image.save(surface, filename)