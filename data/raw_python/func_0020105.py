def cross_validate(self, ax):
        '''
        Performs the cross-validation step.

        '''

        # The CDPP to beat
        cdpp_opt = self.get_cdpp_arr()

        # Loop over all chunks
        for b, brkpt in enumerate(self.breakpoints):

            log.info("Cross-validating chunk %d/%d..." %
                     (b + 1, len(self.breakpoints)))

            # Mask for current chunk
            m = self.get_masked_chunk(b)

            # Mask transits and outliers
            time = self.time[m]
            flux = self.fraw[m]
            ferr = self.fraw_err[m]
            med = np.nanmedian(self.fraw)

            # Setup the GP
            gp = GP(self.kernel, self.kernel_params, white=False)
            gp.compute(time, ferr)

            # The masks
            masks = list(Chunks(np.arange(0, len(time)),
                                len(time) // self.cdivs))

            # The pre-computed matrices
            pre_v = [self.cv_precompute(mask, b) for mask in masks]

            # Initialize with the nPLD solution
            log_lam_opt = np.log10(self.lam[b])
            scatter_opt = self.validation_scatter(
                log_lam_opt, b, masks, pre_v, gp, flux, time, med)
            log.info("Iter 0/%d: " % (self.piter) +
                     "logL = (%s), s = %.3f" %
                     (", ".join(["%.3f" % l for l in log_lam_opt]),
                      scatter_opt))

            # Do `piter` iterations
            for p in range(self.piter):

                # Perturb the initial condition a bit
                log_lam = np.array(
                    np.log10(self.lam[b])) * \
                    (1 + self.ppert * np.random.randn(len(self.lam[b])))
                scatter = self.validation_scatter(
                    log_lam, b, masks, pre_v, gp, flux, time, med)
                log.info("Initializing at: " +
                         "logL = (%s), s = %.3f" %
                         (", ".join(["%.3f" % l for l in log_lam]), scatter))

                # Call the minimizer
                log_lam, scatter, _, _, _, _ = \
                    fmin_powell(self.validation_scatter, log_lam,
                                args=(b, masks, pre_v, gp, flux, time, med),
                                maxfun=self.pmaxf, disp=False,
                                full_output=True)

                # Did it improve the CDPP?
                tmp = np.array(self.lam[b])
                self.lam[b] = 10 ** log_lam
                self.compute()
                cdpp = self.get_cdpp_arr()[b]
                self.lam[b] = tmp
                if cdpp < cdpp_opt[b]:
                    cdpp_opt[b] = cdpp
                    log_lam_opt = log_lam

                # Log it
                log.info("Iter %d/%d: " % (p + 1, self.piter) +
                         "logL = (%s), s = %.3f" %
                         (", ".join(["%.3f" % l for l in log_lam]), scatter))

            # The best solution
            log.info("Found minimum: logL = (%s), s = %.3f" %
                     (", ".join(["%.3f" % l for l in log_lam_opt]),
                      scatter_opt))
            self.lam[b] = 10 ** log_lam_opt

        # We're just going to plot lambda as a function of chunk number
        bs = np.arange(len(self.breakpoints))
        color = ['k', 'b', 'r', 'g', 'y']
        for n in range(self.pld_order):
            ax[0].plot(bs + 1, [np.log10(self.lam[b][n])
                                for b in bs], '.', color=color[n])
            ax[0].plot(bs + 1, [np.log10(self.lam[b][n])
                                for b in bs], '-', color=color[n], alpha=0.25)
        ax[0].set_ylabel(r'$\log\Lambda$', fontsize=5)
        ax[0].margins(0.1, 0.1)
        ax[0].set_xticks(np.arange(1, len(self.breakpoints) + 1))
        ax[0].set_xticklabels([])

        # Now plot the CDPP
        cdpp_arr = self.get_cdpp_arr()
        ax[1].plot(bs + 1, cdpp_arr, 'b.')
        ax[1].plot(bs + 1, cdpp_arr, 'b-', alpha=0.25)
        ax[1].margins(0.1, 0.1)
        ax[1].set_ylabel(r'Scatter (ppm)', fontsize=5)
        ax[1].set_xlabel(r'Chunk', fontsize=5)
        ax[1].set_xticks(np.arange(1, len(self.breakpoints) + 1))