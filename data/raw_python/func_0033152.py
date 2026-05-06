def get_color(self, color):
        '''
        Returns a color to use.

        :param integer/string color:
            Color for the plot. Can be an index for the color from COLORS
            or a key(string) from CNAMES.
        '''
        if color is None:
            color = self.counter
        if isinstance(color, str):
            color = CNAMES[color]
        self.counter = color + 1
        color %= len(COLORS)
        return color