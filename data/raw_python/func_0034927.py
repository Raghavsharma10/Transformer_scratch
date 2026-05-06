def calc_nfw(self, rbins, offsets=None, numTh=200, numRoff=200,
                 numRinner=20, factorRouter=3):
        """Calculates Sigma and DeltaSigma profiles.

        Generates the surface mass density (sigma_nfw attribute of parent
        object) and differential surface mass density (deltasigma_nfw
        attribute of parent object) profiles of each cluster, assuming a
        spherical NFW model. Optionally includes the effect of cluster
        miscentering offsets.

        Parameters
        ----------
        rbins : array_like
            Radial bins (in Mpc) for calculating cluster profiles. Should
            be 1D, optionally with astropy.units of Mpc.
        offsets : array_like, optional
            Parameter describing the width (in Mpc) of the Gaussian
            distribution of miscentering offsets. Should be 1D, optionally
            with astropy.units of Mpc.

        Other Parameters
        -------------------
        numTh : int, optional
            Parameter to pass to SurfaceMassDensity(). Number of bins to
            use for integration over theta, for calculating offset profiles
            (no effect for offsets=None). Default 200.
        numRoff : int, optional
            Parameter to pass to SurfaceMassDensity(). Number of bins to
            use for integration over R_off, for calculating offset profiles
            (no effect for offsets=None). Default 200.
        numRinner : int, optional
            Parameter to pass to SurfaceMassDensity(). Number of bins at
            r < min(rbins) to use for integration over Sigma(<r), for
            calculating DeltaSigma (no effect for Sigma ever, and no effect
            for DeltaSigma if offsets=None). Default 20.
        factorRouter : int, optional
            Parameter to pass to SurfaceMassDensity(). Factor increase over
            number of rbins, at min(r) < r < max(r), of bins that will be
            used at for integration over Sigma(<r), for calculating
            DeltaSigma (no effect for Sigma, and no effect for DeltaSigma
            if offsets=None). Default 3.
        """
        if offsets is None:
            self._sigoffset = np.zeros(self.number) * units.Mpc
        else:
            self._sigoffset = utils.check_units_and_type(offsets, units.Mpc,
                                                         num=self.number)

        self.rbins = utils.check_units_and_type(rbins, units.Mpc)

        rhoc = self._rho_crit.to(units.Msun / units.pc**2 / units.Mpc)
        smd = SurfaceMassDensity(self.rs, self.delta_c, rhoc,
                                 offsets=self._sigoffset,
                                 rbins=self.rbins,
                                 numTh=numTh,
                                 numRoff=numRoff,
                                 numRinner=numRinner,
                                 factorRouter=factorRouter)

        self.sigma_nfw = smd.sigma_nfw()
        self.deltasigma_nfw = smd.deltasigma_nfw()