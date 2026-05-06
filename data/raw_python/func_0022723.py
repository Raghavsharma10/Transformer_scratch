def inner_rect(self):
        """The rectangular area inside the margin, border, and padding.

        Generally widgets should avoid drawing or placing sub-widgets outside
        this rectangle.
        """
        m = self.margin + self._border_width + self.padding
        if not self.border_color.is_blank:
            m += 1
        return Rect((m, m), (self.size[0]-2*m, self.size[1]-2*m))