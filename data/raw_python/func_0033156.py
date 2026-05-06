def lines(self, lines_dict, y='bottom', color='grey', **kwargs):
        '''
        Creates vertical lines in the plot.

        :param lines_dict:
            A dictionary of label, x-coordinate pairs.
        :param y:
            May be 'top', 'bottom' or int.
            The y coordinate of the text-labels.
        :param color color:
            The color of the lines.
        '''
        for l, x in lines_dict.items():
            self.line(x, l, y, color, **kwargs)