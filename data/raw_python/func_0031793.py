def get_synidx(self, cellindex):
        """
        Local function, draw and return synapse locations corresponding
        to a single cell, using a random seed set as
        `POPULATIONSEED` + `cellindex`.


        Parameters
        ----------
        cellindex : int
            Index of cell object.


        Returns
        -------
        synidx : dict
            `LFPy.Cell` compartment indices


        See also
        --------
        Population.get_all_synIdx, Population.fetchSynIdxCell

        """
        #create a cell instance
        cell = self.cellsim(cellindex, return_just_cell=True)


        #local containers
        synidx = {}

        #get synaptic placements and cells from the network,
        #then set spike times,
        for i, X in enumerate(self.X):
            synidx[X] = self.fetchSynIdxCell(cell=cell,
                                             nidx=self.k_yXL[:, i],
                                             synParams=self.synParams.copy())

        return synidx