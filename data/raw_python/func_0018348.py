def update_on_event(self, e):
        """Process a single event."""
        if e.type == QUIT:
            self.running = False

        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                self.running = False

            elif e.key == K_F4 and e.mod & KMOD_ALT:  # Alt+F4 --> quits
                self.running = False

        elif e.type == VIDEORESIZE:
            self.SCREEN_SIZE = e.size
            self.screen = self.new_screen()