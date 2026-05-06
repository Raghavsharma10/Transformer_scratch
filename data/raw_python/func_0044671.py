def rect_helper(x0, y0, x1, y1):
        """Rectangle helper"""
        x0, y0, x1, y1 = force_int(x0, y0, x1, y1)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        return x0, y0, x1, y1