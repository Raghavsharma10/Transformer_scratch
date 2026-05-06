def render(self):
        """Render the screen. Here you must draw everything."""
        self.screen.fill(self.BACKGROUND_COLOR)

        for wid, cond in self._widgets:
            if cond():
                wid.render(self.screen)

        if self.BORDER_COLOR is not None:
            pygame.draw.rect(self.screen, self.BORDER_COLOR, ((0, 0), self.SCREEN_SIZE), 1)

        if self.SHOW_FPS:
            self.fps.render(self.screen)