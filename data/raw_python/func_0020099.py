def run(self):
        '''
        Runs the de-trending step.

        '''

        try:

            # Load raw data
            log.info("Loading target data...")
            self.load_tpf()
            self.mask_planets()
            self.plot_aperture([self.dvs.top_right() for i in range(4)])
            self.init_kernel()
            M = self.apply_mask(np.arange(len(self.time)))
            self.cdppr_arr = self.get_cdpp_arr()
            self.cdpp_arr = np.array(self.cdppr_arr)
            self.cdppv_arr = np.array(self.cdppr_arr)
            self.cdppr = self.get_cdpp()
            self.cdpp = self.cdppr
            self.cdppv = self.cdppr

            log.info("%s (Raw): CDPP = %s" % (self.name, self.cdpps))
            self.plot_lc(self.dvs.left(), info_right='Raw', color='k')

            # Loop
            for n in range(self.pld_order):
                self.lam_idx += 1
                self.get_outliers()
                if n > 0 and self.optimize_gp:
                    self.update_gp()
                self.cross_validate(self.dvs.right(), info='CV%d' % n)
                self.cdpp_arr = self.get_cdpp_arr()
                self.cdppv_arr *= self.cdpp_arr
                self.cdpp = self.get_cdpp()
                self.cdppv = np.nanmean(self.cdppv_arr)
                log.info("%s (%d/%d): CDPP = %s" %
                         (self.name, n + 1, self.pld_order, self.cdpps))
                self.plot_lc(self.dvs.left(), info_right='LC%d' % (
                    n + 1), info_left='%d outliers' % len(self.outmask))

            # Save
            self.finalize()
            self.plot_final(self.dvs.top_left())
            self.plot_info(self.dvs)
            self.save_model()

        except:

            self.exception_handler(self.debug)