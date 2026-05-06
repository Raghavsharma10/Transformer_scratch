def sigma_nfw(self):
        """Calculate NFW surface mass density profile.

        Generate the surface mass density profiles of each cluster halo,
        assuming a spherical NFW model. Optionally includes the effect of
        cluster miscentering offsets, if the parent object was initialized
        with offsets.

        Returns
        ----------
        Quantity
            Surface mass density profiles (ndarray, in astropy.units of
            Msun/pc/pc). Each row corresponds to a single cluster halo.
        """
        def _centered_sigma(self):
            # perfectly centered cluster case

            # calculate f
            bigF = np.zeros_like(self._x)
            f = np.zeros_like(self._x)

            numerator_arg = ((1. / self._x[self._x_small]) +
                             np.sqrt((1. / (self._x[self._x_small]**2)) - 1.))
            denominator = np.sqrt(1. - (self._x[self._x_small]**2))
            bigF[self._x_small] = np.log(numerator_arg) / denominator

            bigF[self._x_big] = (np.arccos(1. / self._x[self._x_big]) /
                                 np.sqrt(self._x[self._x_big]**2 - 1.))

            f = (1. - bigF) / (self._x**2 - 1.)
            f[self._x_one] = 1. / 3.
            if np.isnan(np.sum(f)) or np.isinf(np.sum(f)):
                print('\nERROR: f is not all real\n')

            # calculate & return centered profiles
            if f.ndim == 2:
                sigma = 2. * self._rs_dc_rcrit * f
            else:
                rs_dc_rcrit_4D = self._rs_dc_rcrit.T.reshape(1, 1,
                                                             f.shape[2],
                                                             f.shape[3])
                sigma = 2. * rs_dc_rcrit_4D * f

            return sigma

        def _offset_sigma(self):

            # size of "x" arrays to integrate over
            numRoff = self._numRoff
            numTh = self._numTh

            numRbins = self._nbins
            maxsig = self._sigmaoffset.value.max()

            # inner/outer bin edges
            roff_1D = np.linspace(0., 4. * maxsig, numRoff)
            theta_1D = np.linspace(0., 2. * np.pi, numTh)
            rMpc_1D = self._rbins.value

            # reshape for broadcasting: (numTh,numRoff,numRbins)
            theta = theta_1D.reshape(numTh, 1, 1)
            roff = roff_1D.reshape(1, numRoff, 1)
            rMpc = rMpc_1D.reshape(1, 1, numRbins)

            r_eq13 = np.sqrt(rMpc ** 2 + roff ** 2 -
                             2. * rMpc * roff * np.cos(theta))

            # 3D array r_eq13 -> 4D dimensionless radius (nlens)
            _set_dimensionless_radius(self, radii=r_eq13, integration=True)

            sigma = _centered_sigma(self)
            inner_integrand = sigma.value / (2. * np.pi)

            # INTEGRATE OVER theta
            sigma_of_RgivenRoff = simps(inner_integrand, x=theta_1D, axis=0,
                                        even='first')

            # theta is gone, now dimensions are: (numRoff,numRbins,nlens)
            sig_off_3D = self._sigmaoffset.value.reshape(1, 1, self._nlens)
            roff_v2 = roff_1D.reshape(numRoff, 1, 1)
            PofRoff = (roff_v2 / (sig_off_3D**2) *
                       np.exp(-0.5 * (roff_v2 / sig_off_3D)**2))

            dbl_integrand = sigma_of_RgivenRoff * PofRoff

            # INTEGRATE OVER Roff
            # (integration axis=0 after theta is gone).
            sigma_smoothed = simps(dbl_integrand, x=roff_1D, axis=0,
                                   even='first')

            # reset _x to correspond to input rbins (default)
            _set_dimensionless_radius(self)

            sigma_sm = np.array(sigma_smoothed.T) * units.solMass / units.pc**2

            return sigma_sm

        if self._sigmaoffset is None:
            finalsigma = _centered_sigma(self)
        elif np.abs(self._sigmaoffset).sum() == 0:
            finalsigma = _centered_sigma(self)
        else:
            finalsigma = _offset_sigma(self)
            self._sigma_sm = finalsigma

        return finalsigma