def solve(self, x0, params=(), internal_x0=None, solver=None, attached_solver=None, **kwargs):
        """ Solve with user specified ``solver`` choice.

        Parameters
        ----------
        x0: 1D array of floats
            Guess (subject to ``self.post_processors``)
        params: 1D array_like of floats
            Parameters (subject to ``self.post_processors``)
        internal_x0: 1D array of floats
            When given it overrides (processed) ``x0``. ``internal_x0`` is not
            subject to ``self.post_processors``.
        solver: str or callable or None or iterable of such
            if str: uses _solve_``solver``(\*args, \*\*kwargs).
            if ``None``: chooses from PYNEQSYS_SOLVER environment variable.
            if iterable: chain solving.
        attached_solver: callable factory
            Invokes: solver = attached_solver(self).

        Returns
        -------
        array:
            solution vector (post-processed by self.post_processors)
        dict:
            info dictionary containing 'success', 'nfev', 'njev' etc.

        Examples
        --------
        >>> neqsys = NeqSys(2, 2, lambda x, p: [
        ...     (x[0] - x[1])**p[0]/2 + x[0] - 1,
        ...     (x[1] - x[0])**p[0]/2 + x[1]
        ... ])
        >>> x, sol = neqsys.solve([1, 0], [3], solver=(None, 'mpmath'))
        >>> assert sol['success']
        >>> print(x)
        [0.841163901914009663684741869855]
        [0.158836098085990336315258130144]

        """
        if not isinstance(solver, (tuple, list)):
            solver = [solver]
        if not isinstance(attached_solver, (tuple, list)):
            attached_solver = [attached_solver] + [None]*(len(solver) - 1)
        _x0, self.internal_params = self.pre_process(x0, params)
        for solv, attached_solv in zip(solver, attached_solver):
            if internal_x0 is not None:
                _x0 = internal_x0
            elif self.internal_x0_cb is not None:
                _x0 = self.internal_x0_cb(x0, params)

            nfo = self._get_solver_cb(solv, attached_solv)(_x0, **kwargs)
            _x0 = nfo['x'].copy()
        self.internal_x = _x0
        x0 = self.post_process(self.internal_x, self.internal_params)[0]
        return x0, nfo