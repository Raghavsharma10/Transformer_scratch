def plot_transit_model(self, show=True, fold=None, ax=None):
        '''
        Plot the light curve de-trended with a join instrumental + transit
        model with the best fit transit model overlaid. The transit model
        should be specified using the :py:obj:`transit_model` attribute
        and should be an instance or list of instances of
        :py:class:`everest.transit.TransitModel`.

        :param bool show: Show the plot, or return the `fig, ax` instances? \
               Default `True`
        :param str fold: The name of the planet/transit model on which to \
               fold. If only one model is present, can be set to \
               :py:obj:`True`. Default :py:obj:`False` \
               (does not fold the data).
        :param ax: A `matplotlib` axis instance to use for plotting. \
               Default :py:obj:`None`

        '''

        if self.transit_model is None:
            raise ValueError("No transit model provided!")
        if self.transit_depth is None:
            self.compute()
        if fold is not None:
            if (fold is True and len(self.transit_model) > 1) or \
               (type(fold) is not str):
                raise Exception(
                    "Kwarg `fold` should be the name of the transit " +
                    "model on which to fold the data.")
            if fold is True:
                # We are folding on the first index of `self.transit_model`
                fold = 0
            elif type(fold) is str:
                # Figure out the index of the transit model on which to fold
                fold = np.argmax(
                    [fold == tm.name for tm in self.transit_model])
            log.info('Plotting the transit model folded ' +
                     'on transit model index %d...' % fold)
        else:
            log.info('Plotting the transit model...')

        # Set up axes
        if ax is None:
            if fold is not None:
                fig, ax = pl.subplots(1, figsize=(8, 5))
            else:
                fig, ax = pl.subplots(1, figsize=(13, 6))
            fig.canvas.set_window_title('EVEREST Light curve')
        else:
            fig = pl.gcf()

        # Set up some stuff
        if self.cadence == 'sc':
            ms = 2
        else:
            ms = 4

        # Fold?
        if fold is not None:
            times = self.transit_model[fold].params.get('times', None)
            if times is not None:
                time = self.time - \
                    [times[np.argmin(np.abs(ti - times))] for ti in self.time]
                t0 = times[0]
            else:
                t0 = self.transit_model[fold].params.get('t0', 0.)
                period = self.transit_model[fold].params.get('per', 10.)
                time = (self.time - t0 - period / 2.) % period - period / 2.
            dur = 0.01 * \
                len(np.where(self.transit_model[fold](
                    np.linspace(t0 - 0.5, t0 + 0.5, 100)) < 0)[0])
        else:
            time = self.time
            ax.plot(self.apply_mask(time), self.apply_mask(self.flux),
                    ls='none', marker='.', color='k', markersize=ms, alpha=0.5)
            ax.plot(time[self.outmask], self.flux[self.outmask],
                    ls='none', marker='.', color='k', markersize=ms, alpha=0.5)
            ax.plot(time[self.transitmask], self.flux[self.transitmask],
                    ls='none', marker='.', color='k', markersize=ms, alpha=0.5)

        # Plot the transit + GP model
        med = np.nanmedian(self.apply_mask(self.flux))
        transit_model = \
            med * np.sum([depth * tm(self.time)
                          for tm, depth in zip(self.transit_model,
                                               self.transit_depth)], axis=0)
        gp = GP(self.kernel, self.kernel_params, white=False)
        gp.compute(self.apply_mask(self.time), self.apply_mask(self.fraw_err))
        y, _ = gp.predict(self.apply_mask(
            self.flux - transit_model) - med, self.time)
        if fold is not None:
            flux = (self.flux - y) / med
            ax.plot(self.apply_mask(time), self.apply_mask(flux),
                    ls='none', marker='.', color='k', markersize=ms, alpha=0.5)
            ax.plot(time[self.outmask], flux[self.outmask], ls='none',
                    marker='.', color='k', markersize=ms, alpha=0.5)
            ax.plot(time[self.transitmask], flux[self.transitmask],
                    ls='none', marker='.', color='k', markersize=ms, alpha=0.5)
            hires_time = np.linspace(-5 * dur, 5 * dur, 1000)
            hires_transit_model = 1 + \
                self.transit_depth[fold] * \
                self.transit_model[fold](hires_time + t0)
            ax.plot(hires_time, hires_transit_model, 'r-', lw=1, alpha=1)
        else:
            flux = self.flux
            y += med
            y += transit_model
            ax.plot(time, y, 'r-', lw=1, alpha=1)

        # Plot the bad data points
        bnmask = np.array(
            list(set(np.concatenate([self.badmask, self.nanmask]))), dtype=int)
        bmask = [i for i in self.badmask if i not in self.nanmask]
        ax.plot(time[bmask], flux[bmask], 'r.', markersize=ms, alpha=0.25)

        # Appearance
        ax.set_ylabel('EVEREST Flux', fontsize=18)
        ax.margins(0.01, 0.1)
        if fold is not None:
            ax.set_xlabel('Time From Transit Center (days)', fontsize=18)
            ax.set_xlim(-3 * dur, 3 * dur)
        else:
            ax.set_xlabel('Time (%s)' % self._mission.TIMEUNITS, fontsize=18)
            for brkpt in self.breakpoints[:-1]:
                ax.axvline(time[brkpt], color='r', ls='--', alpha=0.25)
            ax.get_yaxis().set_major_formatter(Formatter.Flux)

        # Get y lims that bound most of the flux
        if fold is not None:
            lo = np.min(hires_transit_model)
            pad = 1.5 * (1 - lo)
            ylim = (lo - pad, 1 + pad)
        else:
            f = np.delete(flux, bnmask)
            N = int(0.995 * len(f))
            hi, lo = f[np.argsort(f)][[N, -N]]
            pad = (hi - lo) * 0.1
            ylim = (lo - pad, hi + pad)
        ax.set_ylim(ylim)

        # Indicate off-axis outliers
        for i in np.where(flux < ylim[0])[0]:
            if i in bmask:
                color = "#ffcccc"
            else:
                color = "#ccccff"
            ax.annotate('', xy=(time[i], ylim[0]), xycoords='data',
                        xytext=(0, 15), textcoords='offset points',
                        arrowprops=dict(arrowstyle="-|>", color=color,
                        alpha=0.5))
        for i in np.where(flux > ylim[1])[0]:
            if i in bmask:
                color = "#ffcccc"
            else:
                color = "#ccccff"
            ax.annotate('', xy=(time[i], ylim[1]), xycoords='data',
                        xytext=(0, -15), textcoords='offset points',
                        arrowprops=dict(arrowstyle="-|>", color=color,
                        alpha=0.5))

        if show:
            pl.show()
            pl.close()
        else:
            return fig, ax