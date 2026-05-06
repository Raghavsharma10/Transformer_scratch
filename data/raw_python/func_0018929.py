def update(self):
        """Determine the total number of AR coefficients.

        >>> from hydpy.models.arma import *
        >>> parameterstep('1d')
        >>> responses(((1., 2.), (1.,)), th_3=((1.,), (1., 2., 3.)))
        >>> derived.ar_order.update()
        >>> derived.ar_order
        ar_order(2, 1)
        """
        responses = self.subpars.pars.control.responses
        self.shape = len(responses)
        self(responses.ar_orders)