def getLVstats(self, *args):
        """Returns I/O stats for LV.
        
        @param args: Two calling conventions are implemented:
                     - Passing two parameters vg and lv.
                     - Passing only one parameter in 'vg-lv' format.  
        @return:     Dict of stats.
        
        """
        if not len(args) in (1, 2):
            raise TypeError("The getLVstats must be called with either "
                            "one or two arguments.")
        if self._vgTree is None:
            self._initDMinfo()
        if len(args) == 1:
            dmdev = self._mapLVname2dm.get(args[0])
        else:
            dmdev = self._mapLVtuple2dm.get(args)
        if dmdev is not None:
            return self.getDevStats(dmdev)
        else:
            return None