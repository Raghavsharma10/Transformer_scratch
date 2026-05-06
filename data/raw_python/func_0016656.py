def get_solver(self, name=None, refresh=False, **filters):
        """Load the configuration for a single solver.

        Makes a blocking web call to `{endpoint}/solvers/remote/{solver_name}/`, where `{endpoint}`
        is a URL configured for the client, and returns a :class:`.Solver` instance
        that can be used to submit sampling problems to the D-Wave API and retrieve results.

        Args:
            name (str):
                ID of the requested solver. ``None`` returns the default solver.
                If default solver is not configured, ``None`` returns the first available
                solver in ``Client``'s class (QPU/software/base).

            **filters (keyword arguments, optional):
                Dictionary of filters over features this solver has to have. For a list of
                feature names and values, see: :meth:`~dwave.cloud.client.Client.get_solvers`.

            order_by (callable/str, default='id'):
                Solver sorting key function (or :class:`Solver` attribute name).
                By default, solvers are sorted by ID/name.

            refresh (bool):
                Return solver from cache (if cached with ``get_solvers()``),
                unless set to ``True``.

        Returns:
            :class:`.Solver`

        Examples:
            This example creates two solvers for a client instantiated from
            a local system's auto-detected default configuration file, which configures
            a connection to a D-Wave resource that provides two solvers. The first
            uses the default solver, the second explicitly selects another solver.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> client.get_solvers()   # doctest: +SKIP
            [Solver(id='2000Q_ONLINE_SOLVER1'), Solver(id='2000Q_ONLINE_SOLVER2')]
            >>> solver1 = client.get_solver()    # doctest: +SKIP
            >>> solver2 = client.get_solver(name='2000Q_ONLINE_SOLVER2')    # doctest: +SKIP
            >>> solver1.id  # doctest: +SKIP
            '2000Q_ONLINE_SOLVER1'
            >>> solver2.id   # doctest: +SKIP
            '2000Q_ONLINE_SOLVER2'
            >>> # code that uses client
            >>> client.close() # doctest: +SKIP

        """
        _LOGGER.debug("Requested a solver that best matches feature filters=%r", filters)

        # backward compatibility: name as the first feature
        if name is not None:
            filters.setdefault('name', name)

        # in absence of other filters, config/env solver filters/name are used
        if not filters and self.default_solver:
            filters = self.default_solver

        # get the first solver that satisfies all filters
        try:
            _LOGGER.debug("Fetching solvers according to filters=%r", filters)
            return self.get_solvers(refresh=refresh, **filters)[0]
        except IndexError:
            raise SolverNotFoundError("Solver with the requested features not available")