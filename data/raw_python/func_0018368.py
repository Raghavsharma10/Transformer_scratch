def render(self, surf):
        """Render the button"""

        if self.clicked:
            icon = self.icon_pressed
        else:
            icon = self.icon

        surf.blit(icon, self)