def make_plot(self):
        """This method creates the waterfall plot.

        """

        # sets levels of main contour plot
        colors1 = ['None', 'darkblue', 'blue', 'deepskyblue', 'aqua',
                   'greenyellow', 'orange', 'red', 'darkred']

        if len(self.contour_vals) > len(colors1) + 1:
            raise AttributeError("Reduce number of contours.")

        # produce filled contour of SNR
        sc = self.axis.contourf(self.xvals[0], self.yvals[0], self.zvals[0],
                                levels=np.asarray(self.contour_vals), colors=colors1)

        self.colorbar.setup_colorbars(sc)

        # check for user desire to show separate contour line
        if self.snr_contour_value is not None:
            self.axis.contour(self.xvals[0], self.yvals[0], self.zvals[0],
                              np.array([self.snr_contour_value]),
                              colors='white', linewidths=1.5, linestyles='dashed')

        return