def led_matrix(self, surface):
        """
        Transforms the input surface into an LED matrix (1 pixel = 1 LED)
        """
        scale = self._led_on.get_width()
        w, h = self._input_size
        pix = self._pygame.PixelArray(surface)
        img = self._pygame.Surface((w * scale, h * scale))

        for y in range(h):
            for x in range(w):
                led = self._led_on if pix[x, y] & 0xFFFFFF > 0 else self._led_off
                img.blit(led, (x * scale, y * scale))

        return img