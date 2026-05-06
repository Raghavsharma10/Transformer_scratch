def px_to_pt(self, px):
        """Convert a size in pxel to a size in points."""
        if px < 200:
            pt = self.PX_TO_PT[px]
        else:
            pt = int(floor((px - 1.21) / 1.332))

        return pt