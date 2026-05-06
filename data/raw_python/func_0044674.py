def filled_rectangle(self, x0, y0, x1, y1):
        """Draw a filled rectangle"""
        x0, y0, x1, y1 = self.rect_helper(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.point(x, y, self.color)