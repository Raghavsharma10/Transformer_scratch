def from_callback(cls, cb, transf_cbs, nx, nparams=0, pre_adj=None,
                      **kwargs):
        """ Generate a TransformedSys instance from a callback

        Parameters
        ----------
        cb : callable
            Should have the signature ``cb(x, p, backend) -> list of exprs``.
            The callback ``cb`` should return *untransformed* expressions.
        transf_cbs : pair or iterable of pairs of callables
            Callables for forward- and backward-transformations. Each
            callable should take a single parameter (expression) and
            return a single expression.
        nx : int
            Number of unkowns.
        nparams : int
            Number of parameters.
        pre_adj : callable, optional
            To tweak expression prior to transformation. Takes a
            sinlge argument (expression) and return a single argument
            rewritten expression.
        \\*\\*kwargs :
            Keyword arguments passed on to :class:`TransformedSys`. See also
            :class:`SymbolicSys` and :class:`pyneqsys.NeqSys`.

        Examples
        --------
        >>> import sympy as sp
        >>> transformed = TransformedSys.from_callback(lambda x, p, be: [
        ...     x[0]*x[1] - p[0],
        ...     be.exp(-x[0]) + be.exp(-x[1]) - p[0]**-2
        ... ], (sp.log, sp.exp), 2, 1)
        ...


        """
        be = Backend(kwargs.pop('backend', None))
        x = be.real_symarray('x', nx)
        p = be.real_symarray('p', nparams)
        try:
            transf = [(transf_cbs[idx][0](xi),
                       transf_cbs[idx][1](xi))
                      for idx, xi in enumerate(x)]
        except TypeError:
            transf = zip(_map2(transf_cbs[0], x), _map2(transf_cbs[1], x))
        try:
            exprs = cb(x, p, be)
        except TypeError:
            exprs = _ensure_3args(cb)(x, p, be)
        return cls(x, _map2l(pre_adj, exprs), transf, p, backend=be, **kwargs)