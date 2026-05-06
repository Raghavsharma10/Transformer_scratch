def solve(self, x0, params=(), internal_x0=None, solver=None,
              conditional_maxiter=20, initial_conditions=None, **kwargs):
        """ Solve the problem (systems of equations)

        Parameters
        ----------
        x0 : array
            Guess.
        params : array
            See :meth:`NeqSys.solve`.
        internal_x0 : array
            See :meth:`NeqSys.solve`.
        solver : str or callable or iterable of such.
            See :meth:`NeqSys.solve`.
        conditional_maxiter : int
            Maximum number of switches between conditions.
        initial_conditions : iterable of bools
            Corresponding conditions to ``x0``
        \\*\\*kwargs :
            Keyword arguments passed on to :meth:`NeqSys.solve`.

        """
        if initial_conditions is not None:
            conds = initial_conditions
        else:
            conds = self.get_conds(x0, params, initial_conditions)
        idx, nfev, njev = 0, 0, 0
        while idx < conditional_maxiter:
            neqsys = self.neqsys_factory(conds)
            x0, info = neqsys.solve(x0, params, internal_x0, solver, **kwargs)
            if idx == 0:
                internal_x0 = None
            nfev += info['nfev']
            njev += info.get('njev', 0)
            new_conds = self.get_conds(x0, params, conds)
            if new_conds == conds:
                break
            else:
                conds = new_conds
            idx += 1
        if idx == conditional_maxiter:
            raise Exception("Solving failed, conditional_maxiter reached")
        self.internal_x = info['x']
        self.internal_params = neqsys.internal_params
        result = {
            'x': info['x'],
            'success': info['success'],
            'conditions': conds,
            'nfev': nfev,
            'njev': njev,
        }
        if 'fun' in info:
            result['fun'] = info['fun']
        return x0, result