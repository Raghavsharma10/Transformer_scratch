def update(self):
        """Update |KD2| based on |EQD2| and |TInd|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> eqd2(0.1)
        >>> tind.value = 10.0
        >>> derived.kd2.update()
        >>> derived.kd2
        kd2(1.0)
        """
        con = self.subpars.pars.control
        self(con.eqd2*con.tind)