def set_spacing(self, space):
        """Set the figure spacing.

        Sets whether in general there is space between subplots.
        If all axes are shared, this can be `tight`. Default in code is `wide`.

        The main difference is the tick labels extend to the ends if space==`wide`.
        If space==`tight`, the edge tick labels are cut off for clearity.

        Args:
            space (str): Sets spacing for subplots. Either `wide` or `tight`.

        """
        self.figure.spacing = space
        if 'subplots_adjust_kwargs' not in self.figure.__dict__:
            self.figure.subplots_adjust_kwargs = {}
        if space == 'wide':
            self.figure.subplots_adjust_kwargs['hspace'] = 0.3
            self.figure.subplots_adjust_kwargs['wspace'] = 0.3
        else:
            self.figure.subplots_adjust_kwargs['hspace'] = 0.0
            self.figure.subplots_adjust_kwargs['wspace'] = 0.0

        return