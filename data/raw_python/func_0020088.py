def cross_validate(self, ax, info=''):
        '''
        Cross-validate to find the optimal value of :py:obj:`lambda`.

        :param ax: The current :py:obj:`matplotlib.pyplot` axis instance to \
               plot the cross-validation results.
        :param str info: The label to show in the bottom right-hand corner \
               of the plot. Default `''`

        '''

        # Loop over all chunks
        ax = np.atleast_1d(ax)
        for b, brkpt in enumerate(self.breakpoints):

            log.info("Cross-validating chunk %d/%d..." %
                     (b + 1, len(self.breakpoints)))
            med_training = np.zeros_like(self.lambda_arr)
            med_validation = np.zeros_like(self.lambda_arr)

            # Mask for current chunk
            m = self.get_masked_chunk(b)

            # Check that we have enough data
            if len(m) < 3 * self.cdivs:
                self.cdppv_arr[b] = np.nan
                self.lam[b][self.lam_idx] = 0.
                log.info(
                    "Insufficient data to run cross-validation on this chunk.")
                continue

            # Mask transits and outliers
            time = self.time[m]
            flux = self.fraw[m]
            ferr = self.fraw_err[m]
            med = np.nanmedian(flux)

            # The precision in the validation set
            validation = [[] for k, _ in enumerate(self.lambda_arr)]

            # The precision in the training set
            training = [[] for k, _ in enumerate(self.lambda_arr)]

            # Setup the GP
            gp = GP(self.kernel, self.kernel_params, white=False)
            gp.compute(time, ferr)

            # The masks
            masks = list(Chunks(np.arange(0, len(time)),
                                len(time) // self.cdivs))

            # Loop over the different masks
            for i, mask in enumerate(masks):

                log.info("Section %d/%d..." % (i + 1, len(masks)))

                # Pre-compute (training set)
                pre_t = self.cv_precompute([], b)

                # Pre-compute (validation set)
                pre_v = self.cv_precompute(mask, b)

                # Iterate over lambda
                for k, lam in enumerate(self.lambda_arr):

                    # Update the lambda matrix
                    self.lam[b][self.lam_idx] = lam

                    # Training set
                    model = self.cv_compute(b, *pre_t)
                    training[k].append(
                        self.fobj(flux - model, med, time, gp, mask))

                    # Validation set
                    model = self.cv_compute(b, *pre_v)
                    validation[k].append(
                        self.fobj(flux - model, med, time, gp, mask))

            # Finalize
            training = np.array(training)
            validation = np.array(validation)
            for k, _ in enumerate(self.lambda_arr):

                # Take the mean
                med_validation[k] = np.nanmean(validation[k])
                med_training[k] = np.nanmean(training[k])

            # Compute best model
            i = self.optimize_lambda(validation)
            v_best = med_validation[i]
            t_best = med_training[i]
            self.cdppv_arr[b] = v_best / t_best
            self.lam[b][self.lam_idx] = self.lambda_arr[i]
            log.info("Found optimum solution at log(lambda) = %.1f." %
                     np.log10(self.lam[b][self.lam_idx]))

            # Plotting: There's not enough space in the DVS to show the
            # cross-val results for more than three light curve segments.
            if len(self.breakpoints) <= 3:

                # Plotting hack: first x tick will be -infty
                lambda_arr = np.array(self.lambda_arr)
                lambda_arr[0] = 10 ** (np.log10(lambda_arr[1]) - 3)

                # Plot cross-val
                for n in range(len(masks)):
                    ax[b].plot(np.log10(lambda_arr),
                               validation[:, n], 'r-', alpha=0.3)

                ax[b].plot(np.log10(lambda_arr),
                           med_training, 'b-', lw=1., alpha=1)
                ax[b].plot(np.log10(lambda_arr),
                           med_validation, 'r-', lw=1., alpha=1)
                ax[b].axvline(np.log10(self.lam[b][self.lam_idx]),
                              color='k', ls='--', lw=0.75, alpha=0.75)
                ax[b].axhline(v_best, color='k', ls='--', lw=0.75, alpha=0.75)
                ax[b].set_ylabel(r'Scatter (ppm)', fontsize=5)
                hi = np.max(validation[0])
                lo = np.min(training)
                rng = (hi - lo)
                ax[b].set_ylim(lo - 0.15 * rng, hi + 0.15 * rng)
                if rng > 2:
                    ax[b].get_yaxis().set_major_formatter(Formatter.CDPP)
                    ax[b].get_yaxis().set_major_locator(
                        MaxNLocator(4, integer=True))
                elif rng > 0.2:
                    ax[b].get_yaxis().set_major_formatter(Formatter.CDPP1F)
                    ax[b].get_yaxis().set_major_locator(MaxNLocator(4))
                else:
                    ax[b].get_yaxis().set_major_formatter(Formatter.CDPP2F)
                    ax[b].get_yaxis().set_major_locator(MaxNLocator(4))

                # Fix the x ticks
                xticks = [np.log10(lambda_arr[0])] + list(np.linspace(
                    np.log10(lambda_arr[1]), np.log10(lambda_arr[-1]), 6))
                ax[b].set_xticks(xticks)
                ax[b].set_xticklabels(['' for x in xticks])
                pad = 0.01 * \
                    (np.log10(lambda_arr[-1]) - np.log10(lambda_arr[0]))
                ax[b].set_xlim(np.log10(lambda_arr[0]) - pad,
                               np.log10(lambda_arr[-1]) + pad)
                ax[b].annotate('%s.%d' % (info, b), xy=(0.02, 0.025),
                               xycoords='axes fraction',
                               ha='left', va='bottom', fontsize=7, alpha=0.25,
                               fontweight='bold')

        # Finally, compute the model
        self.compute()

        # Tidy up
        if len(ax) == 2:
            ax[0].xaxis.set_ticks_position('top')
        for axis in ax[1:]:
            axis.spines['top'].set_visible(False)
            axis.xaxis.set_ticks_position('bottom')

        if len(self.breakpoints) <= 3:

            # A hack to mark the first xtick as -infty
            labels = ['%.1f' % x for x in xticks]
            labels[0] = r'$-\infty$'
            ax[-1].set_xticklabels(labels)
            ax[-1].set_xlabel(r'Log $\Lambda$', fontsize=5)

        else:

            # We're just going to plot lambda as a function of chunk number
            bs = np.arange(len(self.breakpoints))
            ax[0].plot(bs + 1, [np.log10(self.lam[b][self.lam_idx])
                                for b in bs], 'r.')
            ax[0].plot(bs + 1, [np.log10(self.lam[b][self.lam_idx])
                                for b in bs], 'r-', alpha=0.25)
            ax[0].set_ylabel(r'$\log\Lambda$', fontsize=5)
            ax[0].margins(0.1, 0.1)
            ax[0].set_xticks(np.arange(1, len(self.breakpoints) + 1))
            ax[0].set_xticklabels([])

            # Now plot the CDPP and approximate validation CDPP
            cdpp_arr = self.get_cdpp_arr()
            cdppv_arr = self.cdppv_arr * cdpp_arr
            ax[1].plot(bs + 1, cdpp_arr, 'b.')
            ax[1].plot(bs + 1, cdpp_arr, 'b-', alpha=0.25)
            ax[1].plot(bs + 1, cdppv_arr, 'r.')
            ax[1].plot(bs + 1, cdppv_arr, 'r-', alpha=0.25)
            ax[1].margins(0.1, 0.1)
            ax[1].set_ylabel(r'Scatter (ppm)', fontsize=5)
            ax[1].set_xlabel(r'Chunk', fontsize=5)
            if len(self.breakpoints) < 15:
                ax[1].set_xticks(np.arange(1, len(self.breakpoints) + 1))
            else:
                ax[1].set_xticks(np.arange(1, len(self.breakpoints) + 1, 2))