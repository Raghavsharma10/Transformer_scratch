def set_fig_y_label(self, ylabel, **kwargs):
        """Set overall figure y.

        Set label for y axis on overall figure. This is not for a specific plot.
        It will place the label on the figure at the left with a call to ``fig.text``.

        Args:
            ylabel (str): ylabel for entire figure.

        Keyword Arguments:
            x/y (float, optional): The x/y location of the text in figure coordinates.
                Defaults are 0.45 for x and 0.02 for y.
            horizontalalignment/ha (str, optional): The horizontal alignment of
                the text relative to (x, y). Optionas are 'center', 'left', or 'right'.
                Default is 'center'.
            verticalalignment/va (str, optional): The vertical alignment of the text
                relative to (x, y). Optionas are 'top', 'center', 'bottom',
                or 'baseline'. Default is 'top'.
            fontsize/size (int): The font size of the text. Default is 20.
            rotation (float or str): Rotation of label. Options are angle in degrees,
                `horizontal`, or `vertical`. Default is `horizontal`.
            Note: Other kwargs are available.
                See https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.figtext

        """
        prop_default = {
            'x': 0.45,
            'y': 0.02,
            'fontsize': 20,
            'rotation': 'horizontal',
            'ha': 'center',
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        self._set_fig_label('y', ylabel, **kwargs)
        return