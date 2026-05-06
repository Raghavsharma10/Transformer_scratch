def display(self, image):
        """
        Takes a :py:mod:`PIL.Image` and renders it to a pygame display surface.
        """
        assert(image.size == self.size)
        self._last_image = image

        image = self.preprocess(image)
        self._clock.tick(self._fps)
        self._pygame.event.pump()

        if self._abort():
            self._pygame.quit()
            sys.exit()

        surface = self.to_surface(image, alpha=self._contrast)
        if self._screen is None:
            self._screen = self._pygame.display.set_mode(surface.get_size())
        self._screen.blit(surface, (0, 0))
        self._pygame.display.flip()