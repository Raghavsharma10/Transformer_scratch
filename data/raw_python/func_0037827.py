def design(self, max_stimuli=-1, max_inhibitors=-1, max_experiments=10, relax=False, configure=None):
        """
        Finds all optimal experimental designs using up to :attr:`max_experiments` experiments, such that each experiment has
        up to :attr:`max_stimuli` stimuli and :attr:`max_inhibitors` inhibitors. Each optimal experimental design is appended in the
        attribute :attr:`designs` as an instance of :class:`caspo.core.clamping.ClampingList`.

        Example::

            >>> from caspo import core, design
            >>> networks = core.LogicalNetworkList.from_csv('behaviors.csv')
            >>> setup = core.Setup.from_json('setup.json')

            >>> designer = design.Designer(networks, setup)
            >>> designer.design(3, 2)

            >>> for i,d in enumerate(designer.designs):
            ...     f = 'design-%s' % i
            ...     d.to_csv(f, stimuli=self.setup.stimuli, inhibitors=self.setup.inhibitors)



        Parameters
        ----------
        max_stimuli : int
            Maximum number of stimuli per experiment

        max_inhibitors : int
            Maximum number of inhibitors per experiment

        max_experiments : int
            Maximum number of experiments per design

        relax : boolean
            Whether to relax the full-pairwise networks discrimination (True) or not (False).
            If relax equals True, the number of experiments per design is fixed to :attr:`max_experiments`

        configure : callable
            Callable object responsible of setting clingo configuration
        """
        self.designs = []

        args = ['-c maxstimuli=%s' % max_stimuli, '-c maxinhibitors=%s' % max_inhibitors, '-Wno-atom-undefined']

        clingo = gringo.Control(args)
        clingo.conf.solve.opt_mode = 'optN'
        if configure is not None:
            configure(clingo.conf)

        clingo.add("base", [], self.instance)
        clingo.load(self.encodings['design'])

        clingo.ground([("base", [])])

        if relax:
            parts = [("step", [step]) for step in xrange(1, max_experiments+1)]
            parts.append(("diff", [max_experiments + 1]))
            clingo.ground(parts)
            ret = clingo.solve(on_model=self.__save__)
        else:
            step, ret = 0, gringo.SolveResult.UNKNOWN
            while step <= max_experiments and ret != gringo.SolveResult.SAT:
                parts = []
                parts.append(("check", [step]))
                if step > 0:
                    clingo.release_external(gringo.Fun("query", [step-1]))
                    parts.append(("step", [step]))
                    clingo.cleanup_domains()

                clingo.ground(parts)
                clingo.assign_external(gringo.Fun("query", [step]), True)
                ret, step = clingo.solve(on_model=self.__save__), step + 1

        self.stats['time_optimum'] = clingo.stats['time_solve']
        self.stats['time_enumeration'] = clingo.stats['time_total']

        self._logger.info("%s optimal experimental designs found in %.4fs", len(self.designs), self.stats['time_enumeration'])