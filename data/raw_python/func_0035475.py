def make_plot(self):
        """Make the horizon plot.

        """

        self.get_contour_values()
        # sets levels of main contour plot
        colors1 = ['blue', 'green', 'red', 'purple', 'orange',
                   'gold', 'magenta']

        # set contour value. Default is SNR_CUT.
        self.snr_contour_value = (self.SNR_CUT if self.snr_contour_value is None
                                  else self.snr_contour_value)

        # plot contours
        for j in range(len(self.zvals)):
            hz = self.axis.contour(self.xvals[j], self.yvals[j],
                                   self.zvals[j], np.array([self.snr_contour_value]),
                                   colors=colors1[j], linewidths=1., linestyles='solid')

            # plot invisible lines for purpose of creating a legend
            if self.legend_labels != []:
                # plot a curve off of the grid with same color for legend label.
                self.axis.plot([0.1, 0.2], [0.1, 0.2], color=colors1[j],
                               label=self.legend_labels[j])

        if self.add_legend:
            self.axis.legend(**self.legend_kwargs)

        return