def hit(self, pt):
        """Find the view (self, child, or None) under the point `pt`."""

        if self.hidden or not self._enabled:
            return None

        if not self.frame.collidepoint(pt):
            return None

        local_pt = (pt[0] - self.frame.topleft[0],
                    pt[1] - self.frame.topleft[1])

        for child in reversed(self.children):   # front to back
            hit_view = child.hit(local_pt)
            if hit_view is not None:
                return hit_view

        return self