def set_line_width(self, width=1.):
        """Set line width
    
        Parameters
        ----------
        width : float
            The line width.
        """
        width = float(width)
        if width < 0:
            raise RuntimeError('Cannot have width < 0')
        self.glir.command('FUNC', 'glLineWidth', width)