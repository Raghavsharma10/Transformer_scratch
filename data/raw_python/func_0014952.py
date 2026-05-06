def compose(*funcs):
    """Compose `funcs` to a single function.

    >>> compose(operator.abs, operator.add)(-2,-3)
    5
    >>> compose()('nada')
    'nada'
    >>> compose(sorted, set, partial(filter, None))(range(3)[::-1]*2)
    [1, 2]
    """
    # slightly optimized for most common cases and hence verbose
    if len(funcs) == 2: f0,f1=funcs; return lambda *a,**kw: f0(f1(*a,**kw))
    elif len(funcs) == 3: f0,f1,f2=funcs; return lambda *a,**kw: f0(f1(f2(*a,**kw)))
    elif len(funcs) == 0: return lambda x:x     # XXX single kwarg
    elif len(funcs) == 1: return funcs[0]
    else:
        def composed(*args,**kwargs):
            y = funcs[-1](*args,**kwargs)
            for f in funcs[:0:-1]: y = f(y)
            return y
        return composed