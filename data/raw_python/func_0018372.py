def render(self, display):
        """Renders the bar on the display"""

        # the bar
        bar_rect = pygame.Rect(0, 0, self.width, self.height // 3)
        bar_rect.center = self.center
        display.fill(self.bg_color, bar_rect)

        # the cursor
        circle(display, (self.value_px, self.centery), self.height // 2, self.color)

        # the value
        if self.show_val:
            self.text_val.render(display)