def update(self):
        """Update |QFactor| based on |FT| and the current simulation step size.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('1d')
        >>> ft(10.0)
        >>> derived.qfactor.update()
        >>> derived.qfactor
        qfactor(0.115741)
        """
        con = self.subpars.pars.control
        self(con.ft*1000./self.simulationstep.seconds)