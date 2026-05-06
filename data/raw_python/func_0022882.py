def set_blend_equation(self, mode_rgb, mode_alpha=None):
        """Specify the equation for RGB and alpha blending
    
        Parameters
        ----------
        mode_rgb : str
            Mode for RGB.
        mode_alpha : str | None
            Mode for Alpha. If None, ``mode_rgb`` is used.
    
        Notes
        -----
        See ``set_blend_equation`` for valid modes.
        """
        mode_alpha = mode_rgb if mode_alpha is None else mode_alpha
        self.glir.command('FUNC', 'glBlendEquationSeparate', 
                          mode_rgb, mode_alpha)