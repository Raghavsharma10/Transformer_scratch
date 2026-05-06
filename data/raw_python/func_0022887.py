def set_color_mask(self, red, green, blue, alpha):
        """Toggle writing of frame buffer color components
    
        Parameters
        ----------
        red : bool
            Red toggle.
        green : bool
            Green toggle.
        blue : bool
            Blue toggle.
        alpha : bool
            Alpha toggle.
        """
        self.glir.command('FUNC', 'glColorMask', bool(red), bool(green), 
                          bool(blue), bool(alpha))