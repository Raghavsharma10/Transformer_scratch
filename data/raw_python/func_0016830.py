def viable_dim_config(bytes_available, arrays, template,
        dim_ord, nsolvers=1):
    """
    Returns the number of timesteps possible, given the registered arrays
    and a memory budget defined by bytes_available

    Arguments
    ----------------
    bytes_available : int
        The memory budget, or available number of bytes
        for solving the problem.
    arrays : list
        List of dictionaries describing the arrays
    template : dict
        Dictionary containing key-values that will be used
        to replace any string representations of dimensions
        and types. slvr.template_dict() will return something
        suitable.
    dim_ord : list
        list of dimension string names that the problem should be
        subdivided by. e.g. ['ntime', 'nbl', 'nchan'].
        Multple dimensions can be reduced simultaneously using
        the following syntax 'nbl&na'. This is mostly useful for
        the baseline-antenna equivalence.
    nsolvers : int
        Number of solvers to budget for. Defaults to one.

    Returns
    ----------
    A tuple (boolean, dict). The boolean is True if the problem
    can fit within the supplied budget, False otherwise.
    THe dictionary contains the reduced dimensions as key and
    the reduced size as value.
    e.g. (True, { 'time' : 1, 'nbl' : 1 })

    For a dim_ord = ['ntime', 'nbl', 'nchan'], this method will try and fit
    a ntime x nbl x nchan problem into the available number of bytes.
    If this is not possible, it will first set ntime=1, and then try fit an
    1 x nbl x nchan problem into the budget, then a 1 x 1 x nchan
    problem.

    One can specify reductions for specific dimensions.
    For e.g. ['ntime=20', 'nbl=1&na=2', 'nchan=50%']

    will reduce ntime to 20, but no lower. nbl=1&na=2 sets
    both nbl and na to 1 and 2 in the same operation respectively.
    nchan=50\% will continuously halve the nchan dimension
    until it reaches a value of 1.
    """

    if not isinstance(dim_ord, list):
        raise TypeError('dim_ord should be a list')

    # Don't accept non-negative memory budgets
    if bytes_available < 0:
        bytes_available = 0

    modified_dims = {}
    T = template.copy()

    bytes_used = dict_array_bytes_required(arrays, T)*nsolvers

    # While more bytes are used than are available, set
    # dimensions to one in the order specified by the
    # dim_ord argument.
    while bytes_used > bytes_available:
        try:
            dims = dim_ord.pop(0)
            montblanc.log.debug('Applying reduction {s}. '
                'Bytes available: {a} used: {u}'.format(
                    s=dims,
                    a=fmt_bytes(bytes_available),
                    u=fmt_bytes(bytes_used)))
            dims = dims.strip().split('&')
        except IndexError:
            # No more dimensions available for reducing
            # the problem size. Unable to fit the problem
            # within the specified memory budget
            return False, modified_dims

        # Can't fit everything into memory,
        # Lower dimensions and re-evaluate
        for dim in dims:
            match = re.match(__DIM_REDUCTION_RE, dim)

            if not match:
                raise ValueError(
                    "{d} is an invalid dimension reduction string "
                    "Valid strings are for e.g. "
                    "'ntime', 'ntime=20' or 'ntime=20%'"
                        .format(d=dim))

            dim_name = match.group('name')
            dim_value = match.group('value')
            dim_percent = match.group('percent')
            dim_value = 1 if dim_value is None else int(dim_value)

            # Attempt reduction by a percentage
            if dim_percent == '%':
                dim_value = int(T[dim_name] * int(dim_value) / 100.0)
                if dim_value < 1:
                    # This can't be reduced any further
                    dim_value = 1
                else:
                    # Allows another attempt at reduction
                    # by percentage on this dimension
                    dim_ord.insert(0, dim)

            # Apply the dimension reduction
            if T[dim_name] > dim_value:
                modified_dims[dim_name] = dim_value
                T[dim_name] = dim_value
            else:
                montblanc.log.info(('Ignored reduction of {d} '
                    'of size {s} to {v}. ').format(
                        d=dim_name, s=T[dim_name], v=dim_value))

        bytes_used = dict_array_bytes_required(arrays, T)*nsolvers

    return True, modified_dims