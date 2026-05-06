def learn(self, fit=0, size=0, configure=None):
        """
        Learns all (nearly) optimal logical networks with give fitness and size tolerance.
        The first optimum logical network found is saved in the attribute :attr:`optimum` while
        all enumerated logical networks are saved in the attribute :attr:`networks`.

        Example::

            >>> from caspo import core, learn

            >>> graph = core.Graph.read_sif('pkn.sif')
            >>> dataset = core.Dataset('dataset.csv', 30)
            >>> zipped = graph.compress(dataset.setup)

            >>> learner = learn.Learner(zipped, dataset, 2, 'round', 100)
            >>> learner.learn(0.02, 1)

            >>> learner.networks.to_csv('networks.csv')

        Parameters
        ----------
        fit : float
            Fitness tolerance, e.g., use 0.1 for 10% tolerance with respect to the optimum

        size : int
            Size tolerance with respect to the optimum

        configure : callable
            Callable object responsible of setting a custom clingo configuration
        """
        encodings = ['guess', 'fixpoint', 'rss']
        if self.optimum is None:
            clingo = self.__get_clingo__(encodings + ['opt'])
            if configure is not None:
                configure(clingo.conf)

            clingo.ground([("base", [])])
            clingo.solve(on_model=self.__keep_last__)

            self.stats['time_optimum'] = clingo.stats['time_total']
            self._logger.info("Optimum logical network learned in %.4fs", self.stats['time_optimum'])

            tuples = (f.args() for f in self._last)
            self.optimum = core.LogicalNetwork.from_hypertuples(self.hypergraph, tuples)

        predictions = self.optimum.predictions(self.dataset.clampings, self.dataset.readouts.columns).values

        readouts = self.dataset.readouts.values
        pos = ~np.isnan(readouts)

        rss = np.sum((np.vectorize(self.discrete)(readouts[pos]) - predictions[pos]*self.factor)**2)


        self.stats['optimum_mse'] = mean_squared_error(readouts[pos], predictions[pos])
        self.stats['optimum_size'] = self.optimum.size

        self._logger.info("Optimum logical networks has MSE %.4f and size %s", self.stats['optimum_mse'], self.stats['optimum_size'])

        self.networks.reset()

        args = ['-c maxrss=%s' % int(rss + rss*fit), '-c maxsize=%s' % (self.optimum.size + size)]

        clingo = self.__get_clingo__(encodings + ['enum'], args)
        clingo.conf.solve.models = '0'
        if configure is not None:
            configure(clingo.conf)

        clingo.ground([("base", [])])
        clingo.solve(on_model=self.__save__)

        self.stats['time_enumeration'] = clingo.stats['time_total']
        self._logger.info("%s (nearly) optimal logical networks learned in %.4fs", len(self.networks), self.stats['time_enumeration'])