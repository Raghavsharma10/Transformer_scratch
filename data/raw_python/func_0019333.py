def update(self):
        """Update |AbsFHRU| based on |FT| and |FHRU|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(2)
        >>> lnk(ACKER)
        >>> ft(100.0)
        >>> fhru(0.2, 0.8)
        >>> derived.absfhru.update()
        >>> derived.absfhru
        absfhru(20.0, 80.0)
        """
        control = self.subpars.pars.control
        self(control.ft*control.fhru)