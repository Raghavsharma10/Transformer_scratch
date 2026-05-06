def rectangle(self, x0, y0, x1, y1):
        """Draw a rectangle"""
        x0, y0, x1, y1 = self.rect_helper(x0, y0, x1, y1)
        self.polyline([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])