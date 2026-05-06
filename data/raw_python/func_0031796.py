def insert_all_synapses(self, cellindex, cell):

        """
        Insert all synaptic events from all presynaptic layers on
        cell object with index `cellindex`.


        Parameters
        ----------
        cellindex : int
            cell index in the population.
        cell : `LFPy.Cell` instance
            Postsynaptic target cell.


        Returns
        -------
        None


        See also
        --------
        Population.insert_synapse

        """
        for i, X in enumerate(self.X): #range(self.k_yXL.shape[1]):
            synParams = self.synParams
            synParams.update({
                'weight' : self.J_yX[i],
                'tau' : self.tau_yX[i],
                })
            for j in range(len(self.synIdx[cellindex][X])):
                if self.synDelays is not None:
                    synDelays = self.synDelays[cellindex][X][j]
                else:
                    synDelays = None
                self.insert_synapses(cell = cell,
                                cellindex = cellindex,
                                synParams = synParams,
                                idx = self.synIdx[cellindex][X][j],
                                X=X,
                                SpCell = self.SpCells[cellindex][X][j],
                                synDelays = synDelays)