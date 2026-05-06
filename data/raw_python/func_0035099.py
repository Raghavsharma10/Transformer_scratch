def subplots_adjust(self, **kwargs):
        """Adjust subplot spacing and dimensions.

        Adjust bottom, top, right, left, width in between plots, and height in between plots
        with a call to ``plt.subplots_adjust``.

        See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
        for more information.

        Keyword Arguments:
            bottom (float, optional): Sets position of bottom of subplots in figure coordinates.
                Default is 0.1.
            top (float, optional): Sets position of top of subplots in figure coordinates.
                Default is 0.85.
            left (float, optional): Sets position of left edge of subplots in figure coordinates.
                Default is 0.12.
            right (float, optional): Sets position of right edge of subplots in figure coordinates.
                Default is 0.79.
            wspace (float, optional): The amount of width reserved for space between subplots,
               It is expressed as a fraction of the average axis width. Default is 0.3.
            hspace (float, optional): The amount of height reserved for space between subplots,
               It is expressed as a fraction of the average axis width. Default is 0.3.

        """
        prop_default = {
            'bottom': 0.1,
            'top': 0.85,
            'right': 0.9,
            'left': 0.12,
            'hspace': 0.3,
            'wspace': 0.3,
        }

        if 'subplots_adjust_kwargs' in self.figure.__dict__:
            for key, value in self.figure.subplots_adjust_kwargs.items():
                prop_default[key] = value

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        self.figure.subplots_adjust_kwargs = kwargs
        return