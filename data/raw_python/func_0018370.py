def _start(self):
        """Starts checking if the SB is shifted"""

        # TODO : make an update method instead

        last_call = 42
        while self._focus:
            sleep(1 / 100)

            mouse = pygame.mouse.get_pos()
            last_value = self.get()
            self.value_px = mouse[0]

            # we do not need to do anything when it the same value
            if self.get() == last_value:
                continue

            if last_call + self.interval / 1000 < time():
                last_call = time()
                self.func(self.get())