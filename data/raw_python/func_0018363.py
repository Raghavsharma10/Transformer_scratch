def _front_delta(self):
        """Return the offset of the colored part."""
        if self.flags & self.NO_MOVE:
            return Separator(0, 0)

        if self.clicked and self.hovered:  # the mouse is over the button
            delta = 2

        elif self.hovered and not self.flags & self.NO_HOVER:
            delta = 0

        else:
            delta = 0

        return Separator(delta, delta)