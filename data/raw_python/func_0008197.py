def display(self, image):
        """
        Takes an image, scales it according to the nominated transform, and
        stores it for later building into an animated GIF.
        """
        assert(image.size == self.size)
        self._last_image = image

        image = self.preprocess(image)
        surface = self.to_surface(image, alpha=self._contrast)
        rawbytes = self._pygame.image.tostring(surface, "RGB", False)
        im = Image.frombytes("RGB", surface.get_size(), rawbytes)
        self._images.append(im)

        self._count += 1
        logger.debug("Recording frame: {0}".format(self._count))

        if self._max_frames and self._count >= self._max_frames:
            sys.exit(0)