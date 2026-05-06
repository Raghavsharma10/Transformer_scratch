def copy_rect(self, x0, y0, x1, y1, dx, dy, destination):
        """Copy (blit) a rectangle onto another part of the image"""
        x0, y0, x1, y1 = self.rect_helper(x0, y0, x1, y1)
        dx, dy = force_int(dx, dy)

        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                d = destination._offset(dx + x - x0, dy + y - y0)
                o = self._offset(x, y)
                destination.canvas[d:d + 4] = self.canvas[o:o + 4]