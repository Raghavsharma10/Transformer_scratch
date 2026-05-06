def update(self):
        """Determine the number of response functions.

        >>> from hydpy.models.arma import *
        >>> parameterstep('1d')
        >>> responses(((1., 2.), (1.,)), th_3=((1.,), (1., 2., 3.)))
        >>> derived.nmb.update()
        >>> derived.nmb
        nmb(2)

        Note that updating parameter `nmb` sets the shape of the flux
        sequences |QPIn|, |QPOut|, |QMA|, and |QAR| automatically.

        >>> fluxes.qpin
        qpin(nan, nan)
        >>> fluxes.qpout
        qpout(nan, nan)
        >>> fluxes.qma
        qma(nan, nan)
        >>> fluxes.qar
        qar(nan, nan)
        """
        pars = self.subpars.pars
        responses = pars.control.responses
        fluxes = pars.model.sequences.fluxes
        self(len(responses))
        fluxes.qpin.shape = self.value
        fluxes.qpout.shape = self.value
        fluxes.qma.shape = self.value
        fluxes.qar.shape = self.value