def line(self, x, label=None, y='bottom', color='grey', **kwargs):
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
        super(DiffPlotter, self).line(x, label, y, color, self.ax1, **kwargs)
        super(DiffPlotter, self).line(x, '', 0, color, self.ax2, **kwargs)