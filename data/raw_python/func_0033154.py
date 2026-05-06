def plots(self, series_list, label_list, colors=None):
        '''
        Plots all the series from the list.
        The assumption is that all of the series share the same index.

        :param list series_list:
            A list of series which should be plotted
        :param list label_list:
            A list of labels corresponding to the series
        :params list list_of_colors:
            A list of colors to use.
        '''
        colors = colors or range(len(series_list))
        for series, label, color in zip(series_list, label_list, colors):
            self.plot(series=series, label=label, color=color)