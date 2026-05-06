def filter(self, networks):
        """
        Returns a new experimental setup restricted to species present in the given list of networks

        Parameters
        ----------
        networks : :class:`caspo.core.logicalnetwork.LogicalNetworkList`
            List of logical networks

        Returns
        -------
        caspo.core.setup.Setup
            The restricted experimental setup
        """
        cues = self.stimuli + self.inhibitors
        active_cues = set()
        active_readouts = set()
        for clause, var in networks.mappings:
            active_cues = active_cues.union((l for (l, s) in clause if l in cues))
            if var in self.readouts:
                active_readouts.add(var)

        return Setup(active_cues.intersection(self.stimuli), active_cues.intersection(self.inhibitors), active_readouts)