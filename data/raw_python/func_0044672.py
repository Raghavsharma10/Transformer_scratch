def vertical_gradient(self, x0, y0, x1, y1, start, end):
        """Draw a vertical gradient"""
        x0, y0, x1, y1 = self.rect_helper(x0, y0, x1, y1)
        grad = gradient_list(start, end, y1 - y0)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.point(x, y, grad[y - y0])