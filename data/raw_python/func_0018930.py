def update(self):
        """Determine all AR coefficients.

        >>> from hydpy.models.arma import *
        >>> parameterstep('1d')
        >>> responses(((1., 2.), (1.,)), th_3=((1.,), (1., 2., 3.)))
        >>> derived.ar_coefs.update()
        >>> derived.ar_coefs
        ar_coefs([[1.0, 2.0],
                  [1.0, nan]])

        Note that updating parameter `ar_coefs` sets the shape of the log
        sequence |LogOut| automatically.

        >>> logs.logout
        logout([[nan, nan],
                [nan, nan]])
        """
        pars = self.subpars.pars
        coefs = pars.control.responses.ar_coefs
        self.shape = coefs.shape
        self(coefs)
        pars.model.sequences.logs.logout.shape = self.shape