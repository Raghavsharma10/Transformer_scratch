def get_border_widths(self):
        """Return border width for each side top, left, bottom, right."""
        if type(self.border_widths) is int:   # uniform size
            return [self.border_widths] * 4
        return self.border_widths