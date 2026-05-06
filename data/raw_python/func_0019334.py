def update(self):
        """Update |KInz| based on |HInz| and |LAI|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(2)
        >>> hinz(0.2)
        >>> lai.acker_jun = 1.0
        >>> lai.vers_dec = 2.0
        >>> derived.kinz.update()
        >>> from hydpy import round_
        >>> round_(derived.kinz.acker_jun)
        0.2
        >>> round_(derived.kinz.vers_dec)
        0.4
        """
        con = self.subpars.pars.control
        self(con.hinz*con.lai)