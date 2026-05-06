def _rotate_vector(x, y, x2, y2, x1, y1):
        """
        rotate x,y vector over x2-x1, y2-y1 angle
        """
        angle = atan2(y2 - y1, x2 - x1)
        cos_rad = cos(angle)
        sin_rad = sin(angle)
        return cos_rad * x + sin_rad * y, -sin_rad * x + cos_rad * y