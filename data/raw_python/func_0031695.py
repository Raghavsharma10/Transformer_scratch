def get_xy(self, xlim, fraction=1.):
        """
        Get pairs of node units and spike trains on specific time interval.
        
        
        Parameters
        ----------
        xlim : list of floats
            Spike time interval, e.g., [0., 1000.].
        fraction : float in [0, 1.]
            If less than one, sample a fraction of nodes in random order.
        
        
        Returns
        -------
        x : dict
            In `x` key-value entries are population name and neuron spike times.
        y : dict
            Where in `y` key-value entries are population name and neuron gid number.

        """
        x = {}
        y = {}

        for X, nodes in self.nodes.items():
            x[X] = np.array([])
            y[X] = np.array([])

            if fraction != 1:
                nodes = np.random.permutation(nodes)[:int(nodes.size*fraction)]
                nodes.sort()

            spiketimes = self.dbs[X].select_neurons_interval(nodes, T=xlim)
            i = 0
            for times in spiketimes:
                x[X] = np.r_[x[X], times]
                y[X] = np.r_[y[X], np.zeros(times.size) + nodes[i]]
                i += 1
                
        return x, y