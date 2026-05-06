def line(self, x, label=None, y='bottom', color='grey', ax=None, **kwargs):
        '''
        Creates a vertical line in the plot.

        :param x:
            The x coordinate of the line. Should be in the same units
            as the x-axis.
        :param string label:
            The label to be displayed.
        :param y:
            May be 'top', 'bottom' or int.
            The y coordinate of the text-label.
        :param color color:
            The color of the line.
        '''
        if ax is None:
            ax = plt
            y0, y1 = ax.ylim()
        else:
            y0, y1 = ax.get_ylim()
        ax.axvline(x, color=color, **kwargs)
        if label is not None:
            verticalalignment = 'bottom'
            if y == 'bottom':
                y = y0 + (y1 - y0) / 25.
            if y == 'top':
                verticalalignment = 'top'
                y = y0 + (y1 - y0) * 24 / 25.
            ax.annotate('\n' + label, (x, y), rotation=90,
                        verticalalignment=verticalalignment)