def update(self):
        """Update |QFactor| based on |Area| and the current simulation
        step size.

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> area(50.0)
        >>> derived.qfactor.update()
        >>> derived.qfactor
        qfactor(1.157407)
        """
        self(self.subpars.pars.control.area*1000. /
             self.subpars.qfactor.simulationstep.seconds)