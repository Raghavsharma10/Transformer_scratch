def deltasigma_nfw(self):
        """Calculate NFW differential surface mass density profile.

        Generate the differential surface mass density profiles of each cluster
        halo, assuming a spherical NFW model. Optionally includes the effect of
        cluster miscentering offsets, if the parent object was initialized
        with offsets.

        Returns
        ----------
        Quantity
            Differential surface mass density profiles (ndarray, in
            astropy.units of Msun/pc/pc). Each row corresponds to a single
            cluster halo.
        """
        def _centered_dsigma(self):
            # calculate g

            firstpart = np.zeros_like(self._x)
            secondpart = np.zeros_like(self._x)
            g = np.zeros_like(self._x)

            small_1a = 4. / self._x[self._x_small]**2
            small_1b = 2. / (self._x[self._x_small]**2 - 1.)
            small_1c = np.sqrt(1. - self._x[self._x_small]**2)
            firstpart[self._x_small] = (small_1a + small_1b) / small_1c

            big_1a = 8. / (self._x[self._x_big]**2 *
                           np.sqrt(self._x[self._x_big]**2 - 1.))
            big_1b = 4. / ((self._x[self._x_big]**2 - 1.)**1.5)
            firstpart[self._x_big] = big_1a + big_1b

            small_2a = np.sqrt((1. - self._x[self._x_small]) /
                               (1. + self._x[self._x_small]))
            secondpart[self._x_small] = np.log((1. + small_2a) /
                                               (1. - small_2a))

            big_2a = self._x[self._x_big] - 1.
            big_2b = 1. + self._x[self._x_big]
            secondpart[self._x_big] = np.arctan(np.sqrt(big_2a / big_2b))

            both_3a = (4. / (self._x**2)) * np.log(self._x / 2.)
            both_3b = 2. / (self._x**2 - 1.)
            g = firstpart * secondpart + both_3a - both_3b

            g[self._x_one] = (10. / 3.) + 4. * np.log(0.5)

            if np.isnan(np.sum(g)) or np.isinf(np.sum(g)):
                print('\nERROR: g is not all real\n', g)

            # calculate & return centered profile
            deltasigma = self._rs_dc_rcrit * g

            return deltasigma

        def _offset_dsigma(self):
            original_rbins = self._rbins.value

            # if offset sigma was already calculated, use it!
            try:
                sigma_sm_rbins = self._sigma_sm
            except AttributeError:
                sigma_sm_rbins = self.sigma_nfw()

            innermost_sampling = 1.e-10  # stable for anything below 1e-5
            inner_prec = self._numRinner
            r_inner = np.linspace(innermost_sampling,
                                  original_rbins.min(),
                                  endpoint=False, num=inner_prec)
            outer_prec = self._factorRouter * self._nbins
            r_outer = np.linspace(original_rbins.min(),
                                  original_rbins.max(),
                                  endpoint=False, num=outer_prec + 1)[1:]
            r_ext_unordered = np.hstack([r_inner, r_outer, original_rbins])
            r_extended = np.sort(r_ext_unordered)

            # set temporary extended rbins, nbins, x, rs_dc_rcrit array
            self._rbins = r_extended * units.Mpc
            self._nbins = self._rbins.shape[0]
            _set_dimensionless_radius(self)  # uses _rbins, _nlens
            rs_dc_rcrit = self._rs * self._delta_c * self._rho_crit
            self._rs_dc_rcrit = rs_dc_rcrit.reshape(self._nlens,
                                                    1).repeat(self._nbins, 1)

            sigma_sm_extended = self.sigma_nfw()
            mean_inside_sigma_sm = np.zeros([self._nlens,
                                             original_rbins.shape[0]])

            for i, r in enumerate(original_rbins):
                index_of_rbin = np.where(r_extended == r)[0][0]
                x = r_extended[0:index_of_rbin + 1]
                y = sigma_sm_extended[:, 0:index_of_rbin + 1] * x

                integral = simps(y, x=x, axis=-1, even='first')

                # average of sigma_sm at r < rbin
                mean_inside_sigma_sm[:, i] = (2. / r**2) * integral

            mean_inside_sigma_sm = mean_inside_sigma_sm * (units.Msun /
                                                           units.pc**2)

            # reset original rbins, nbins, x
            self._rbins = original_rbins * units.Mpc
            self._nbins = self._rbins.shape[0]
            _set_dimensionless_radius(self)
            rs_dc_rcrit = self._rs * self._delta_c * self._rho_crit
            self._rs_dc_rcrit = rs_dc_rcrit.reshape(self._nlens,
                                                    1).repeat(self._nbins, 1)
            self._sigma_sm = sigma_sm_rbins  # reset to original sigma_sm

            dsigma_sm = mean_inside_sigma_sm - sigma_sm_rbins

            return dsigma_sm

        if self._sigmaoffset is None:
            finaldeltasigma = _centered_dsigma(self)
        elif np.abs(self._sigmaoffset).sum() == 0:
            finaldeltasigma = _centered_dsigma(self)
        else:
            finaldeltasigma = _offset_dsigma(self)

        return finaldeltasigma