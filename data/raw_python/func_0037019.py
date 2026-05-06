def single_val(self):
        """return relative error of worst point that might make the data none
        symmetric.
        """
        
        sv_t = self._sv(self._tdsphere)
        sv_p = self._sv(self._tdsphere)
        
        return (sv_t, sv_p)