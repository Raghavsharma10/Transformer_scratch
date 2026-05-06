def plot(self, show=True, plot_raw=True, plot_gp=True,
             plot_bad=True, plot_out=True, plot_cbv=True,
             simple=False):
        '''
        Plots the final de-trended light curve.

        :param bool show: Show the plot or return the `(fig, ax)` instance? \
               Default :py:obj:`True`
        :param bool plot_raw: Show the raw light curve? Default :py:obj:`True`
        :param bool plot_gp: Show the GP model prediction? \
               Default :py:obj:`True`
        :param bool plot_bad: Show and indicate the bad data points? \
               Default :py:obj:`True`
        :param bool plot_out: Show and indicate the outliers? \
               Default :py:obj:`True`
        :param bool plot_cbv: Plot the CBV-corrected light curve? \
               Default :py:obj:`True`. If :py:obj:`False`, plots the \
               de-trended but uncorrected light curve.

        '''

        log.info('Plotting the light curve...')

        # Set up axes
        if plot_raw:
            fig, axes = pl.subplots(2, figsize=(13, 9), sharex=True)
            fig.subplots_adjust(hspace=0.1)
            axes = [axes[1], axes[0]]
            if plot_cbv:
                fluxes = [self.fcor, self.fraw]
            else:
                fluxes = [self.flux, self.fraw]
            labels = ['EVEREST Flux', 'Raw Flux']
        else:
            fig, axes = pl.subplots(1, figsize=(13, 6))
            axes = [axes]
            if plot_cbv:
                fluxes = [self.fcor]
            else:
                fluxes = [self.flux]
            labels = ['EVEREST Flux']
        fig.canvas.set_window_title('EVEREST Light curve')

        # Set up some stuff
        time = self.time
        badmask = self.badmask
        nanmask = self.nanmask
        outmask = self.outmask
        transitmask = self.transitmask
        fraw_err = self.fraw_err
        breakpoints = self.breakpoints
        if self.cadence == 'sc':
            ms = 2
        else:
            ms = 4

        # Get the cdpps
        cdpps = [[self.get_cdpp(self.flux), self.get_cdpp_arr(self.flux)],
                 [self.get_cdpp(self.fraw), self.get_cdpp_arr(self.fraw)]]
        self.cdpp = cdpps[0][0]
        self.cdpp_arr = cdpps[0][1]

        for n, ax, flux, label, c in zip([0, 1], axes, fluxes, labels, cdpps):

            # Initialize CDPP
            cdpp = c[0]
            cdpp_arr = c[1]

            # Plot the good data points
            ax.plot(self.apply_mask(time), self.apply_mask(flux),
                    ls='none', marker='.', color='k', markersize=ms, alpha=0.5)

            # Plot the outliers
            bnmask = np.array(
                list(set(np.concatenate([badmask, nanmask]))), dtype=int)
            bmask = [i for i in self.badmask if i not in self.nanmask]

            def O1(x): return x[outmask]

            def O2(x): return x[bmask]

            def O3(x): return x[transitmask]
            if plot_out:
                ax.plot(O1(time), O1(flux), ls='none', color="#777777",
                        marker='.', markersize=ms, alpha=0.5)
            if plot_bad:
                ax.plot(O2(time), O2(flux), 'r.', markersize=ms, alpha=0.25)
            ax.plot(O3(time), O3(flux), 'b.', markersize=ms, alpha=0.25)

            # Plot the GP
            if n == 0 and plot_gp and self.cadence != 'sc':
                gp = GP(self.kernel, self.kernel_params)
                gp.compute(self.apply_mask(time), self.apply_mask(fraw_err))
                med = np.nanmedian(self.apply_mask(flux))
                y, _ = gp.predict(self.apply_mask(flux) - med, time)
                y += med
                ax.plot(self.apply_mask(time), self.apply_mask(
                    y), 'r-', lw=0.5, alpha=0.5)

            # Appearance
            if n == 0:
                ax.set_xlabel('Time (%s)' %
                              self._mission.TIMEUNITS, fontsize=18)
            ax.set_ylabel(label, fontsize=18)
            for brkpt in breakpoints[:-1]:
                ax.axvline(time[brkpt], color='r', ls='--', alpha=0.25)
            if len(cdpp_arr) == 2:
                ax.annotate('%.2f ppm' % cdpp_arr[0], xy=(0.02, 0.975),
                            xycoords='axes fraction',
                            ha='left', va='top', fontsize=12, color='r',
                            zorder=99)
                ax.annotate('%.2f ppm' % cdpp_arr[1], xy=(0.98, 0.975),
                            xycoords='axes fraction',
                            ha='right', va='top', fontsize=12,
                            color='r', zorder=99)
            elif len(cdpp_arr) < 6:
                for n in range(len(cdpp_arr)):
                    if n > 0:
                        x = (self.time[self.breakpoints[n - 1]] - self.time[0]
                             ) / (self.time[-1] - self.time[0]) + 0.02
                    else:
                        x = 0.02
                    ax.annotate('%.2f ppm' % cdpp_arr[n], xy=(x, 0.975),
                                xycoords='axes fraction',
                                ha='left', va='top', fontsize=10,
                                zorder=99, color='r')
            else:
                ax.annotate('%.2f ppm' % cdpp, xy=(0.02, 0.975),
                            xycoords='axes fraction',
                            ha='left', va='top', fontsize=12,
                            color='r', zorder=99)
            ax.margins(0.01, 0.1)

            # Get y lims that bound 99% of the flux
            f = np.concatenate([np.delete(f, bnmask) for f in fluxes])
            N = int(0.995 * len(f))
            hi, lo = f[np.argsort(f)][[N, -N]]
            pad = (hi - lo) * 0.1
            ylim = (lo - pad, hi + pad)
            ax.set_ylim(ylim)
            ax.get_yaxis().set_major_formatter(Formatter.Flux)

            # Indicate off-axis outliers
            for i in np.where(flux < ylim[0])[0]:
                if i in bmask:
                    color = "#ffcccc"
                    if not plot_bad:
                        continue
                elif i in outmask:
                    color = "#cccccc"
                    if not plot_out:
                        continue
                elif i in nanmask:
                    continue
                else:
                    color = "#ccccff"
                ax.annotate('', xy=(time[i], ylim[0]), xycoords='data',
                            xytext=(0, 15), textcoords='offset points',
                            arrowprops=dict(arrowstyle="-|>", color=color))
            for i in np.where(flux > ylim[1])[0]:
                if i in bmask:
                    color = "#ffcccc"
                    if not plot_bad:
                        continue
                elif i in outmask:
                    color = "#cccccc"
                    if not plot_out:
                        continue
                elif i in nanmask:
                    continue
                else:
                    color = "#ccccff"
                ax.annotate('', xy=(time[i], ylim[1]), xycoords='data',
                            xytext=(0, -15), textcoords='offset points',
                            arrowprops=dict(arrowstyle="-|>", color=color))

        # Show total CDPP improvement
        pl.figtext(0.5, 0.94, '%s %d' % (self._mission.IDSTRING, self.ID),
                   fontsize=18, ha='center', va='bottom')
        pl.figtext(0.5, 0.905,
                   r'$%.2f\ \mathrm{ppm} \rightarrow %.2f\ \mathrm{ppm}$' %
                   (self.cdppr, self.cdpp), fontsize=14,
                   ha='center', va='bottom')

        if show:
            pl.show()
            pl.close()
        else:
            if plot_raw:
                return fig, axes
            else:
                return fig, axes[0]