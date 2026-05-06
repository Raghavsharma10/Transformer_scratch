def update(self):
        """Update |WZ| based on |RelWZ| and |NFk|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(2)
        >>> lnk(ACKER)
        >>> relwz(0.8)
        >>> nfk(100.0, 200.0)
        >>> derived.wz.update()
        >>> derived.wz
        wz(80.0, 160.0)
        """
        con = self.subpars.pars.control
        self(con.relwz*con.nfk)