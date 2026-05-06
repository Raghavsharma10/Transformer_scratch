def set_fig_title(self, title, **kwargs):
        """Set overall figure title.

        Set title for overall figure. This is not for a specific plot.
        It will place the title at the top of the figure with a call to ``fig.suptitle``.

        Args:
            title (str): Figure title.

        Keywork Arguments:
            x/y (float, optional): The x/y location of the text in figure coordinates.
                Defaults are 0.5 for x and 0.98 for y.
            horizontalalignment/ha (str, optional): The horizontal alignment of
                the text relative to (x, y). Optionas are 'center', 'left', or 'right'.
                Default is 'center'.
            verticalalignment/va (str, optional): The vertical alignment of the text
                relative to (x, y). Optionas are 'top', 'center', 'bottom',
                or 'baseline'. Default is 'top'.
            fontsize/size (int, optional): The font size of the text. Default is 20.

        """
        prop_default = {
            'fontsize': 20,
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        self.figure.fig_title = title
        self.figure.fig_title_kwargs = kwargs
        return