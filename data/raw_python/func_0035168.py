def setup_figure(self):
        """Sets up the initial figure on to which every plot is added.

        """

        # declare figure and axes environments
        fig, ax = plt.subplots(nrows=int(self.num_rows),
                               ncols=int(self.num_cols),
                               sharex=self.sharex,
                               sharey=self.sharey)

        fig.set_size_inches(self.figure_width, self.figure_height)

        # create list of ax. Catch error if it is a single plot.
        try:
            ax = ax.ravel()
        except AttributeError:
            ax = [ax]

        # create list of plot types
        self.plot_types = [self.plot_info[str(i)]['plot_type'] for i in range(len(ax))]

        if len(self.plot_types) == 1:
            if self.plot_types[0] not in self.colorbars:
                self.colorbars[self.plot_types[0]] = {'cbar_pos': 5}
            else:
                if 'cbar_pos' not in self.colorbars[self.plot_types[0]]:
                    self.colorbars[self.plot_types[0]]['cbar_pos'] = 5

        # prepare colorbar classes
        self.colorbar_classes = {}
        for plot_type in self.plot_types:
            if plot_type in self.colorbar_classes:
                continue
            if plot_type == 'Horizon':
                self.colorbar_classes[plot_type] = None

            elif plot_type in self.colorbars:
                self.colorbar_classes[plot_type] = FigColorbar(fig, plot_type,
                                                               **self.colorbars[plot_type])

            else:
                self.colorbar_classes[plot_type] = FigColorbar(fig, plot_type)

        # set subplots_adjust settings
        if 'Ratio' in self.plot_types or 'Waterfall':
            self.subplots_adjust_kwargs['right'] = 0.79

        # adjust figure sizes
        fig.subplots_adjust(**self.subplots_adjust_kwargs)

        if 'fig_y_label' in self.__dict__.keys():
            fig.text(self.fig_y_label_x,
                     self.fig_y_label_y,
                     r'{}'.format(self.fig_y_label),
                     **self.fig_y_label_kwargs)

        if 'fig_x_label' in self.__dict__.keys():
            fig.text(self.fig_x_label_x,
                     self.fig_x_label_y,
                     r'{}'.format(self.fig_x_label),
                     **self.fig_x_label_kwargs)

        if 'fig_title' in self.__dict__.keys():
            fig.text(self.fig_title_kwargs['x'],
                     self.fig_title_kwargs['y'],
                     r'{}'.format(self.fig_title),
                     **self.fig_title_kwargs)

        self.fig, self.ax = fig, ax
        return