def render(self, surf):
        """Draw the button on the surface."""
        if not self.flags & self.NO_SHADOW:
            circle(surf, self.center + self._bg_delta, self.width / 2, LIGHT_GREY)
        circle(surf, self.center + self._front_delta, self.width / 2, self._get_color())

        self.text.center = self.center + self._front_delta
        self.text.render(surf)