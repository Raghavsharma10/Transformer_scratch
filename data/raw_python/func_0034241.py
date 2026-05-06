def on_event(self, event):
        """Pygame event processing callback method.

        :param event: Event to process.
        """
        if self.state > 0:
            if event.type == MOUSEBUTTONDOWN:
                key = self.layout.get_key_at(pygame.mouse.get_pos())
                if key is not None:
                    self.on_key_down(key)
            elif event.type == MOUSEBUTTONUP:
                self.on_key_up()
            elif event.type == KEYDOWN:
                value = pygame.key.name(event.key)
                # TODO : Find from layout (consider checking layout key space ?)
            elif event.type == KEYUP:
                value = pygame.key.name(event.key)