def doit(self):
        """Perform a simulation run over the actual simulation time period
        defined by the |Timegrids| object stored in module |pub|."""
        idx_start, idx_end = self.simindices
        self.open_files(idx_start)
        methodorder = self.methodorder
        for idx in printtools.progressbar(range(idx_start, idx_end)):
            for func in methodorder:
                func(idx)
        self.close_files()