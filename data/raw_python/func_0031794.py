def fetchSynIdxCell(self, cell, nidx, synParams):
        """
        Find possible synaptic placements for each cell
        As synapses are placed within layers with bounds determined by
        self.layerBoundaries, it will check this matrix accordingly, and
        use the probabilities from `self.connProbLayer to distribute.

        For each layer, the synapses are placed with probability normalized
        by membrane area of each compartment


        Parameters
        ----------
        cell : `LFPy.Cell` instance
        nidx : numpy.ndarray
            Numbers of synapses per presynaptic population X.
        synParams : which `LFPy.Synapse` parameters to use.


        Returns
        -------
        syn_idx : list
            List of arrays of synapse placements per connection.


        See also
        --------
        Population.get_all_synIdx, Population.get_synIdx, LFPy.Synapse

        """

        #segment indices in each layer is stored here, list of np.array
        syn_idx = []
        #loop over layer bounds, find synapse locations
        for i, zz in enumerate(self.layerBoundaries):
            if nidx[i] == 0:
                syn_idx.append(np.array([], dtype=int))
            else:
                syn_idx.append(cell.get_rand_idx_area_norm(
                                section=synParams['section'],
                                nidx=nidx[i],
                                z_min=zz.min(),
                                z_max=zz.max()).astype('int16'))

        return syn_idx