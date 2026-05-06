def update(self):
        """Update |KI2| based on |EQI2| and |TInd|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> eqi2(1.0)
        >>> tind.value = 10.0
        >>> derived.ki2.update()
        >>> derived.ki2
        ki2(10.0)
        """
        con = self.subpars.pars.control
        self(con.eqi2*con.tind)