def calc_signal_sum(self, measure='LFP'):
        """
        Superimpose each cell's contribution to the compound population signal,
        i.e., the population CSD or LFP


        Parameters
        ----------
        measure : str
            {'LFP', 'CSD'}: Either 'LFP' or 'CSD'.


        Returns
        -------
        numpy.ndarray
            The populations-specific compound signal.

        """
        #compute the total LFP of cells on this RANK
        if self.RANK_CELLINDICES.size > 0:
            for i, cellindex in enumerate(self.RANK_CELLINDICES):
                if i == 0:
                    data = self.output[cellindex][measure]
                else:
                    data += self.output[cellindex][measure]
        else:
            data = np.zeros((len(self.electrodeParams['x']),
                             self.cellParams['tstopms']/self.dt_output + 1),
                dtype=np.float32)

        #container for full LFP on RANK 0
        if RANK == 0:
            DATA = np.zeros_like(data, dtype=np.float32)
        else:
            DATA = None

        #sum to RANK 0 using automatic type discovery with MPI
        COMM.Reduce(data, DATA, op=MPI.SUM, root=0)

        return DATA