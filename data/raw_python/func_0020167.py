def _plot_weights(self, show=True):
        '''
        .. warning:: Untested!

        '''

        # Set up the axes
        fig = pl.figure(figsize=(12, 12))
        fig.subplots_adjust(top=0.95, bottom=0.025, left=0.1, right=0.92)
        fig.canvas.set_window_title(
            '%s %d' % (self._mission.IDSTRING, self.ID))
        ax = [pl.subplot2grid((80, 130), (20 * j, 25 * i), colspan=23,
                              rowspan=18)
              for j in range(len(self.breakpoints) * 2)
              for i in range(1 + 2 * (self.pld_order - 1))]
        cax = [pl.subplot2grid((80, 130),
                               (20 * j, 25 * (1 + 2 * (self.pld_order - 1))),
                               colspan=4, rowspan=18)
               for j in range(len(self.breakpoints) * 2)]
        ax = np.array(ax).reshape(2 * len(self.breakpoints), -1)
        cax = np.array(cax)

        # Check number of segments
        if len(self.breakpoints) > 3:
            log.error('Cannot currently plot weights for light ' +
                      'curves with more than 3 segments.')
            return

        # Loop over all PLD orders and over all chunks
        npix = len(self.fpix[1])
        ap = self.aperture.flatten()
        ncol = 1 + 2 * (len(self.weights[0]) - 1)
        raw_weights = np.zeros(
            (len(self.breakpoints), ncol, self.aperture.shape[0],
             self.aperture.shape[1]), dtype=float)
        scaled_weights = np.zeros(
            (len(self.breakpoints), ncol, self.aperture.shape[0],
             self.aperture.shape[1]), dtype=float)

        # Loop over orders
        for o in range(len(self.weights[0])):
            if o == 0:
                oi = 0
            else:
                oi = 1 + 2 * (o - 1)

            # Loop over chunks
            for b in range(len(self.weights)):

                c = self.get_chunk(b)
                rw_ii = np.zeros(npix)
                rw_ij = np.zeros(npix)
                sw_ii = np.zeros(npix)
                sw_ij = np.zeros(npix)
                X = np.nanmedian(self.X(o, c), axis=0)

                # Compute all sets of pixels at this PLD order, then
                # loop over them and assign the weights to the correct pixels
                sets = np.array(list(multichoose(np.arange(npix).T, o + 1)))
                for i, s in enumerate(sets):
                    if (o == 0) or (s[0] == s[1]):
                        # Not the cross-terms
                        j = s[0]
                        rw_ii[j] += self.weights[b][o][i]
                        sw_ii[j] += X[i] * self.weights[b][o][i]
                    else:
                        # Cross-terms
                        for j in s:
                            rw_ij[j] += self.weights[b][o][i]
                            sw_ij[j] += X[i] * self.weights[b][o][i]

                # Make the array 2D and plot it
                rw = np.zeros_like(ap, dtype=float)
                sw = np.zeros_like(ap, dtype=float)
                n = 0
                for i, a in enumerate(ap):
                    if (a & 1):
                        rw[i] = rw_ii[n]
                        sw[i] = sw_ii[n]
                        n += 1
                raw_weights[b][oi] = rw.reshape(*self.aperture.shape)
                scaled_weights[b][oi] = sw.reshape(*self.aperture.shape)

                if o > 0:
                    # Make the array 2D and plot it
                    rw = np.zeros_like(ap, dtype=float)
                    sw = np.zeros_like(ap, dtype=float)
                    n = 0
                    for i, a in enumerate(ap):
                        if (a & 1):
                            rw[i] = rw_ij[n]
                            sw[i] = sw_ij[n]
                            n += 1
                    raw_weights[b][oi + 1] = rw.reshape(*self.aperture.shape)
                    scaled_weights[b][oi +
                                      1] = sw.reshape(*self.aperture.shape)

        # Plot the images
        log.info('Plotting the PLD weights...')
        rdbu = pl.get_cmap('RdBu_r')
        rdbu.set_bad('k')
        for b in range(len(self.weights)):
            rmax = max([-raw_weights[b][o].min() for o in range(ncol)] +
                       [raw_weights[b][o].max() for o in range(ncol)])
            smax = max([-scaled_weights[b][o].min() for o in range(ncol)] +
                       [scaled_weights[b][o].max() for o in range(ncol)])
            for o in range(ncol):
                imr = ax[2 * b, o].imshow(raw_weights[b][o], aspect='auto',
                                          interpolation='nearest', cmap=rdbu,
                                          origin='lower', vmin=-rmax,
                                          vmax=rmax)
                ims = ax[2 * b + 1, o].imshow(scaled_weights[b][o],
                                              aspect='auto',
                                              interpolation='nearest',
                                              cmap=rdbu, origin='lower',
                                              vmin=-smax, vmax=smax)

            # Colorbars
            def fmt(x, pos):
                a, b = '{:.0e}'.format(x).split('e')
                b = int(b)
                if float(a) > 0:
                    a = r'+' + a
                elif float(a) == 0:
                    return ''
                return r'${} \times 10^{{{}}}$'.format(a, b)
            cbr = pl.colorbar(imr, cax=cax[2 * b], format=FuncFormatter(fmt))
            cbr.ax.tick_params(labelsize=8)
            cbs = pl.colorbar(
                ims, cax=cax[2 * b + 1], format=FuncFormatter(fmt))
            cbs.ax.tick_params(labelsize=8)

        # Plot aperture contours
        def PadWithZeros(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector
        ny, nx = self.aperture.shape
        contour = np.zeros((ny, nx))
        contour[np.where(self.aperture)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode='nearest')
        extent = np.array([-1, nx, -1, ny])
        for axis in ax.flatten():
            axis.contour(highres, levels=[
                         0.5], extent=extent, origin='lower', colors='r',
                         linewidths=1)

            # Check for saturated columns
            for x in range(self.aperture.shape[0]):
                for y in range(self.aperture.shape[1]):
                    if self.aperture[x][y] == AP_SATURATED_PIXEL:
                        axis.fill([y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                                  [x - 0.5, x - 0.5, x + 0.5, x + 0.5],
                                  fill=False, hatch='xxxxx', color='r', lw=0)

            axis.set_xlim(-0.5, nx - 0.5)
            axis.set_ylim(-0.5, ny - 0.5)
            axis.set_xticks([])
            axis.set_yticks([])

        # Labels
        titles = [r'$1^{\mathrm{st}}$',
                  r'$2^{\mathrm{nd}}\ (i = j)$',
                  r'$2^{\mathrm{nd}}\ (i \neq j)$',
                  r'$3^{\mathrm{rd}}\ (i = j)$',
                  r'$3^{\mathrm{rd}}\ (i \neq j)$'] + ['' for i in range(10)]
        for i, axis in enumerate(ax[0]):
            axis.set_title(titles[i], fontsize=12)
        for j in range(len(self.weights)):
            ax[2 * j, 0].text(-0.55, -0.15, r'$%d$' % (j + 1),
                              fontsize=16, transform=ax[2 * j, 0].transAxes)
            ax[2 * j, 0].set_ylabel(r'$w_{ij}$', fontsize=18)
            ax[2 * j + 1,
                0].set_ylabel(r'$\bar{X}_{ij} \cdot w_{ij}$', fontsize=18)

        if show:
            pl.show()
            pl.close()
        else:
            return fig, ax, cax