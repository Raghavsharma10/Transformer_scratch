def setup_plot(self):
        """Set up limits and labels.

        For all plot types, this method is used to setup the basic features of each plot.

        """
        if self.tick_label_fontsize is not None:
            self.x_tick_label_fontsize = self.tick_label_fontsize
            self.y_tick_label_fontsize = self.tick_label_fontsize

        # setup xticks and yticks and limits
        # if logspaced, the log values are used.
        xticks = np.arange(float(self.xlims[0]),
                           float(self.xlims[1])
                           + float(self.dx),
                           float(self.dx))

        yticks = np.arange(float(self.ylims[0]),
                           float(self.ylims[1])
                           + float(self.dy),
                           float(self.dy))

        xlim = [xticks.min(), xticks.max()]
        ylim = [yticks.min(), yticks.max()]

        if self.reverse_x_axis:
                xticks = xticks[::-1]
                xlim = [xticks.max(), xticks.min()]

        if self.reverse_y_axis:
                    yticks = yticks[::-1]
                    ylim = [yticks.max(), yticks.min()]

        self.axis.set_xlim(xlim)
        self.axis.set_ylim(ylim)

        # adjust ticks for spacing. If 'wide' then show all labels, if 'tight' remove end labels.
        if self.spacing == 'wide':
            x_inds = np.arange(len(xticks))
            y_inds = np.arange(len(yticks))
        else:
            # remove end labels
            x_inds = np.arange(1, len(xticks)-1)
            y_inds = np.arange(1, len(yticks)-1)

        self.axis.set_xticks(xticks[x_inds])
        self.axis.set_yticks(yticks[y_inds])

        # set tick labels based on scale
        if self.xscale == 'log':
            self.axis.set_xticklabels([r'$10^{%i}$' % int(i)
                                      for i in xticks[x_inds]], fontsize=self.x_tick_label_fontsize)
        else:
            self.axis.set_xticklabels([r'$%.3g$' % (i)
                                      for i in xticks[x_inds]], fontsize=self.x_tick_label_fontsize)

        if self.yscale == 'log':
            self.axis.set_yticklabels([r'$10^{%i}$' % int(i)
                                      for i in yticks[y_inds]], fontsize=self.y_tick_label_fontsize)
        else:
            self.axis.set_yticklabels([r'$%.3g$' % (i)
                                      for i in yticks[y_inds]], fontsize=self.y_tick_label_fontsize)

        # add grid
        if self.add_grid:
            self.axis.grid(True, linestyle='-', color='0.75')

        # add title
        if 'title' in self.__dict__.keys():
            self.axis.set_title(r'{}'.format(self.title), **self.title_kwargs)

        if 'xlabel' in self.__dict__.keys():
            self.axis.set_xlabel(r'{}'.format(self.xlabel), **self.xlabel_kwargs)

        if 'ylabel' in self.__dict__.keys():
            self.axis.set_ylabel(r'{}'.format(self.ylabel), **self.ylabel_kwargs)
        return