def evalfunc(cls, func, *args, **kwargs):
        """Evaluate a function with error propagation.

        Inputs:
        -------
            ``func``: callable
                this is the function to be evaluated. Should return either a
                number or a np.ndarray.
            ``*args``: other positional arguments of func. Arguments which are
                not instances of `ErrorValue` are taken as constants.

            keyword arguments supported:
                ``NMC``: number of Monte-Carlo steps. If not defined, defaults
                to 1000
                ``exceptions_to_retry``: list of exception types to ignore:
                    if one of these is raised the given MC step is repeated once
                    again. Notice that this might induce an infinite loop!
                    The exception types in this list should be subclasses of
                    ``Exception``.
                ``exceptions_to_skip``: list of exception types to skip: if
                    one of these is raised the given MC step is skipped, never
                    to be repeated. The exception types in this list should be
                    subclasses of ``Exception``.


        Output:
        -------
            ``result``: an `ErrorValue` with the result. The error is estimated
                via a Monte-Carlo approach to Gaussian error propagation.
        """

        def do_random(x):
            if isinstance(x, cls):
                return x.random()
            else:
                return x

        if 'NMC' not in kwargs:
            kwargs['NMC'] = 1000
        if 'exceptions_to_skip' not in kwargs:
            kwargs['exceptions_to_skip'] = []
        if 'exceptions_to_repeat' not in kwargs:
            kwargs['exceptions_to_repeat'] = []
        meanvalue = func(*args)
        # this way we get either a number or a np.array
        stdcollector = meanvalue * 0
        mciters = 0
        while mciters < kwargs['NMC']:
            try:
                # IGNORE:W0142
                stdcollector += (func(*[do_random(a)
                                        for a in args]) - meanvalue) ** 2
                mciters += 1
            except Exception as e:  # IGNORE:W0703
                if any(isinstance(e, etype) for etype in kwargs['exceptions_to_skip']):
                    kwargs['NMC'] -= 1
                elif any(isinstance(e, etype) for etype in kwargs['exceptions_to_repeat']):
                    pass
                else:
                    raise
        return cls(meanvalue, stdcollector ** 0.5 / (kwargs['NMC'] - 1))