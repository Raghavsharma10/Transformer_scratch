def _ufunc_wrapper(ufunc, name=None):
    """
    A function to generate the top level biggus ufunc wrappers.

    """
    if not isinstance(ufunc, np.ufunc):
        raise TypeError('{} is not a ufunc'.format(ufunc))

    ufunc_name = ufunc.__name__
    # Get hold of the masked array equivalent, if it exists.
    ma_ufunc = getattr(np.ma, ufunc_name, None)
    if ufunc.nin == 2 and ufunc.nout == 1:
        func = _dual_input_fn_wrapper('np.{}'.format(ufunc_name), ufunc,
                                      ma_ufunc, name)
    elif ufunc.nin == 1 and ufunc.nout == 1:
        func = _unary_fn_wrapper('np.{}'.format(ufunc_name), ufunc, ma_ufunc,
                                 name)
    else:
        raise ValueError('Unsupported ufunc {!r} with {} input arrays & {} '
                         'output arrays.'.format(ufunc_name, ufunc.nin,
                                                 ufunc.nout))
    return func