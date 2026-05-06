def control(self, size=0, configure=None):
        """
        Finds all inclusion-minimal intervention strategies up to the given size.
        Intervention strategies found are saved in the attribute :attr:`strategies`
        as a :class:`caspo.core.clamping.ClampingList` object instance.

        Example::

            >>> from caspo import core, control

            >>> networks = core.LogicalNetworkList.from_csv('networks.csv')
            >>> scenarios = control.ScenarioList('scenarios.csv')

            >>> controller = control.Controller(networks, scenarios)
            >>> controller.control()

            >>> controller.strategies.to_csv('strategies.csv')

        Parameters
        ----------
        size : int
            Maximum number of intervention per intervention strategy

        configure : callable
            Callable object responsible of setting clingo configuration
        """
        self._strategies = []

        clingo = gringo.Control(['-c maxsize=%s' % size])

        clingo.conf.solve.models = '0'
        if configure:
            def overwrite(args, proxy):
                for i in xrange(args.threads):
                    proxy.solver[i].no_lookback = 'false'
                    proxy.solver[i].heuristic = 'domain'
                    proxy.solver[i].dom_mod = '5,16'

            configure(clingo.conf, overwrite)
        else:
            clingo.conf.solver.no_lookback = 'false'
            clingo.conf.solver.heuristic = 'domain'
            clingo.conf.solver.dom_mod = '5,16'

        clingo.conf.solve.enum_mode = 'domRec'

        clingo.add("base", [], self.instance)
        clingo.load(self.encodings['control'])

        clingo.ground([("base", [])])
        clingo.solve(on_model=self.__save__)

        self.stats['time_optimum'] = clingo.stats['time_solve']
        self.stats['time_enumeration'] = clingo.stats['time_total']

        self._logger.info("%s optimal intervention strategies found in %.4fs", len(self._strategies), self.stats['time_enumeration'])

        self.strategies = core.ClampingList(self._strategies)