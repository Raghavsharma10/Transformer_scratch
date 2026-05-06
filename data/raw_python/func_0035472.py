def make_plot(self):
        """Creates the ratio plot.

        """
        # sets colormap for ratio comparison plot
        cmap = getattr(cm, self.colormap)

        # set values of ratio comparison contour
        normval = 2.0
        num_contours = 40  # must be even
        levels = np.linspace(-normval, normval, num_contours)
        norm = colors.Normalize(-normval, normval)

        # find Loss/Gain contour and Ratio contour
        self.set_comparison()
        diff_out, loss_gain_contour = self.find_difference_contour()

        cmap.set_bad(color='white', alpha=0.001)
        # plot ratio contours

        sc = self.axis.contourf(self.xvals[0], self.yvals[0], diff_out,
                                levels=levels, norm=norm,
                                extend='both', cmap=cmap)

        self.colorbar.setup_colorbars(sc)

        # toggle line contours of orders of magnitude of ratio comparisons
        if self.order_contour_lines:
                self.axis.contour(self.xvals[0], self.yvals[0], diff_out, np.array(
                    [-2.0, -1.0, 1.0, 2.0]), colors='black', linewidths=1.0)

        # plot loss gain contour
        if self.loss_gain_status is True:
            # if there is no loss/gain contours, this will produce an error,
            # so we catch the exception.
            try:
                # make hatching
                cs = self.axis.contourf(self.xvals[0], self.yvals[0],
                                        loss_gain_contour, levels=[-2, -0.5, 0.5, 2], colors='none',
                                        hatches=['x', None, '+'])
                # make loss/gain contour outline
                self.axis.contour(self.xvals[0], self.yvals[0],
                                  loss_gain_contour, 3, colors='black', linewidths=2)

            except ValueError:
                pass

        if self.add_legend:
            loss_patch = Patch(fill=None, label='Loss', hatch='x', linestyle='--', linewidth=2)
            gain_patch = Patch(fill=None, label='Gain', hatch='+', linestyle='-', linewidth=2)
            legend = self.axis.legend(handles=[loss_patch, gain_patch], **self.legend_kwargs)

        return