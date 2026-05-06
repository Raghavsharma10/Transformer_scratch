def classify(self, n_jobs=-1, configure=None):
        """
        Returns input-output behaviors for the list of logical networks in the attribute :attr:`networks`

        Example::

            >>> from caspo import core, classify

            >>> networks = core.LogicalNetworkList.from_csv('networks.csv')
            >>> setup = core.Setup.from_json('setup.json')

            >>> classifier = classify.Classifier(networks, setup)
            >>> behaviors = classifier.classify()

            >>> behaviors.to_csv('behaviors.csv', networks=True)

        n_jobs : int
            Number of jobs to run in parallel. Default to -1 (all cores available)

        configure : callable
            Callable object responsible of setting clingo configuration


        Returns
        -------
        caspo.core.logicalnetwork.LogicalNetworkList
            The list of networks with one representative for each behavior
        """
        start = timeit.default_timer()
        networks = self.networks

        n = len(networks)
        cpu = n_jobs if n_jobs > -1 else mp.cpu_count()

        if cpu > 1:
            lpart = int(np.ceil(n / float(cpu))) if n > cpu else 1
            parts = networks.split(np.arange(lpart, n, lpart))

            behaviors_parts = Parallel(n_jobs=n_jobs)(delayed(__learn_io__)(part, self.setup, configure) for part in parts)
            networks = core.LogicalNetworkList.from_hypergraph(networks.hg)
            for behavior in behaviors_parts:
                networks = networks.concat(behavior)

        behaviors = __learn_io__(networks, self.setup, configure)
        self.stats['time_io'] = timeit.default_timer() - start

        self._logger.info("%s input-output logical behaviors found in %.4fs", len(behaviors), self.stats['time_io'])

        return behaviors