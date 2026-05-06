def blend_rect(self, x0, y0, x1, y1, dx, dy, destination, alpha=0xff):
        """Blend a rectangle onto the image"""
        x0, y0, x1, y1 = self.rect_helper(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                o = self._offset(x, y)
                rgba = self.canvas[o:o + 4]
                rgba[3] = alpha
                destination.point(dx + x - x0, dy + y - y0, rgba)