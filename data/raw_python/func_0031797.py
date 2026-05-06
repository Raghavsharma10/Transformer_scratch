def fetchSpCells(self, nodes, numSyn):
        """
        For N (nodes count) nestSim-cells draw
        POPULATION_SIZE x NTIMES random cell indexes in
        the population in nodes and broadcast these as `SpCell`.

        The returned argument is a list with len = numSyn.size of np.arrays,
        assumes `numSyn` is a list


        Parameters
        ----------
        nodes : numpy.ndarray, dtype=int
            Node # of valid presynaptic neurons.
        numSyn : numpy.ndarray, dtype=int
            # of synapses per connection.


        Returns
        -------
        SpCells : list
            presynaptic network-neuron indices


        See also
        --------
        Population.fetch_all_SpCells
        """
        SpCell = []
        for size in numSyn:
            SpCell.append(np.random.randint(nodes.min(), nodes.max(),
                                            size=size).astype('int32'))
        return SpCell