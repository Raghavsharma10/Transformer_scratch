def update(self):
        """Determine all MA coefficients.

        >>> from hydpy.models.arma import *
        >>> parameterstep('1d')
        >>> responses(((1., 2.), (1.,)), th_3=((1.,), (1., 2., 3.)))
        >>> derived.ma_coefs.update()
        >>> derived.ma_coefs
        ma_coefs([[1.0, nan, nan],
                  [1.0, 2.0, 3.0]])

        Note that updating parameter `ar_coefs` sets the shape of the log
        sequence |LogIn| automatically.

        >>> logs.login
        login([[nan, nan, nan],
               [nan, nan, nan]])
        """
        pars = self.subpars.pars
        coefs = pars.control.responses.ma_coefs
        self.shape = coefs.shape
        self(coefs)
        pars.model.sequences.logs.login.shape = self.shape