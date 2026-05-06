def evaluate(x, temperature):
        """Evaluate the model.

        Parameters
        ----------
        x : number or ndarray
            Wavelengths in Angstrom.

        temperature : number
            Temperature in Kelvin.

        Returns
        -------
        y : number or ndarray
            Blackbody radiation in PHOTLAM per steradian.

        """
        if ASTROPY_LT_2_0:
            from astropy.analytic_functions.blackbody import blackbody_nu
        else:
            from astropy.modeling.blackbody import blackbody_nu

        # Silence Numpy
        old_np_err_cfg = np.seterr(all='ignore')

        wave = np.ascontiguousarray(x) * u.AA
        bbnu_flux = blackbody_nu(wave, temperature)
        bbflux = (bbnu_flux * u.sr).to(
            units.PHOTLAM, u.spectral_density(wave)) / u.sr  # PHOTLAM/sr

        # Restore Numpy settings
        np.seterr(**old_np_err_cfg)

        return bbflux.value