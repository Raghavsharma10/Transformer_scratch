def display(self, image):
        """
        Takes a :py:mod:`PIL.Image` and renders it to the current terminal as
        ASCII-blocks.
        """
        assert(image.size == self.size)
        self._last_image = image

        surface = self.to_surface(self.preprocess(image), alpha=self._contrast)
        rawbytes = self._pygame.image.tostring(surface, "RGB", False)
        image = Image.frombytes("RGB", surface.get_size(), rawbytes)

        scr_width = self._terminal_size()[1]
        scale = float(scr_width) / image.width

        self._CSI('1;1H')  # Move to top/left

        for (fg, bg) in self._generate_art(image, int(image.width * scale), int(image.height * scale)):
            self._CSI('38;5;{0};48;5;{1}m'.format(fg, bg))
            sys.stdout.write('▄')

        self._CSI('0m')
        sys.stdout.flush()