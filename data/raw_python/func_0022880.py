def set_clear_color(self, color='black', alpha=None):
        """Set the screen clear color

        This is a wrapper for gl.glClearColor.

        Parameters
        ----------
        color : str | tuple | instance of Color
            Color to use. See vispy.color.Color for options.
        alpha : float | None
            Alpha to use.
        """
        self.glir.command('FUNC', 'glClearColor', *Color(color, alpha).rgba)