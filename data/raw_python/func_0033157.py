def plot(self, series, series_diff=None, label='', color=None, style=None):
        '''
        :param pandas.Series series:
            The series to be plotted, all values must be positive if stacked
            is True.
        :param pandas.Series series_diff:
            The series representing the diff that will be plotted in the
            bottom part.
        :param string label:
            The label for the series.
        :param integer/string color:
            Color for the plot. Can be an index for the color from COLORS
            or a key(string) from CNAMES.
        :param string style:
            Style forwarded to the plt.plot.
        '''
        color = self.get_color(color)
        if series_diff is None and self.autodiffs:
            series_diff = series.diff()
        if self.stacked:
            series += self.running_sum
            self.ax1.fill_between(series.index, self.running_sum, series,
                                  facecolor=ALPHAS[color])
            self.running_sum = series
            self.ax1.set_ylim(bottom=0, top=int(series.max() * 1.05))
        series.plot(label=label, c=COLORS[color], linewidth=2, style=style,
                    ax=self.ax1)
        if series_diff is not None:
            series_diff.plot(label=label, c=COLORS[color], linewidth=2,
                             style=style, ax=self.ax2)