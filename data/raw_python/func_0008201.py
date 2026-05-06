def display(self, image):
        """
        Takes a :py:mod:`PIL.Image` and renders it to the current terminal as
        ASCII-art.
        """
        assert(image.size == self.size)
        self._last_image = image

        surface = self.to_surface(self.preprocess(image), alpha=self._contrast)
        rawbytes = self._pygame.image.tostring(surface, "RGB", False)
        image = Image.frombytes("RGB", surface.get_size(), rawbytes)

        scr_width = self._stdscr.getmaxyx()[1]
        scale = float(scr_width) / image.width

        self._stdscr.erase()
        self._stdscr.move(0, 0)
        try:
            for (ch, color) in self._generate_art(image, int(image.width * scale), int(image.height * scale)):
                self._stdscr.addstr(ch, curses.color_pair(color))

        except curses.error:
            # End of screen reached
            pass

        self._stdscr.refresh()