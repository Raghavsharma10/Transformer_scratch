def set_marker_color(self, color='#3ea0e4', edgecolor='k'):
        """ set the marker color used in the plot
        :param color: matplotlib color (ie 'r', '#000000')
        """
        # TODO allow a colour set per another variable
        self._marker_color = color
        self._edge_color = edgecolor