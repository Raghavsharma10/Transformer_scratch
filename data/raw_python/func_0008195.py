def to_surface(self, image, alpha=1.0):
        """
        Converts a :py:mod:`PIL.Image` into a :class:`pygame.Surface`,
        transforming it according to the ``transform`` and ``scale``
        constructor arguments.
        """
        assert(0.0 <= alpha <= 1.0)
        if alpha < 1.0:
            im = image.convert("RGBA")
            black = Image.new(im.mode, im.size, "black")
            im = Image.blend(black, im, alpha)
        else:
            im = image.convert("RGB")

        mode = im.mode
        size = im.size
        data = im.tobytes()
        del im

        surface = self._pygame.image.fromstring(data, size, mode)
        return self._transform(surface)