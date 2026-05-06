def set_fig_size(self, width, height=None):
        """Set the figure size in inches.

        Sets the figure size with a call to fig.set_size_inches.
        Default in code is 8 inches for each.

        Args:
            width (float): Dimensions for figure width in inches.
            height (float, optional): Dimensions for figure height in inches. Default is None.

        """
        self.figure.figure_width = width
        self.figure.figure_height = height
        return