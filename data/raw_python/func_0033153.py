def plot(self, series, label='', color=None, style=None):
        '''
        Wrapper around plot.

        :param pandas.Series series:
            The series to be plotted, all values must be positive if stacked
            is True.
        :param string label:
            The label for the series.
        :param integer/string color:
            Color for the plot. Can be an index for the color from COLORS
            or a key(string) from CNAMES.
        :param string style:
            Style forwarded to the plt.plot.
        '''
        color = self.get_color(color)
        if self.stacked:
            series += self.running_sum
            plt.fill_between(series.index, self.running_sum, series,
                             facecolor=ALPHAS[color])
            self.running_sum = series
            plt.gca().set_ylim(bottom=0, top=int(series.max() * 1.05))
        series.plot(label=label, c=COLORS[color], linewidth=2, style=style)