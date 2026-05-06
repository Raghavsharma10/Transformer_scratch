def render(self, surf):
        """Render the button on a surface."""
        pos, size = self.topleft, self.size

        if not self.flags & self.NO_SHADOW:
            if self.flags & self.NO_ROUNDING:
                pygame.draw.rect(surf, LIGHT_GREY, (pos + self._bg_delta, size))
            else:
                roundrect(surf, (pos + self._bg_delta, size), LIGHT_GREY + (100,), 5)

        if self.flags & self.NO_ROUNDING:
            pygame.draw.rect(surf, self._get_color(), (pos + self._front_delta, size))
        else:
            roundrect(surf, (pos + self._front_delta, size), self._get_color(), 5)

        self.text.center = self.center + self._front_delta
        self.text.render(surf)