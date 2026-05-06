def list_updater(*args):
    """
    Decorate a function with named lists into updater for transact.
    
    :params \*args: parameter list sizes. -1 means all other items. None means a single item instead of a list.
                    only one -1 is allowed.
    """
    neg_index = [i for v,i in izip(args, itertools.count()) if v is not None and v < 0]
    if len(neg_index) > 1:
        raise ValueError("Cannot use negative values more than once")
    if not neg_index:
        slice_list = []
        size = 0
        for arg in args:
            if arg is None:
                slice_list.append(size)
                size += 1
            else:
                slice_list.append(slice(size, size + arg))
                size += arg
    else:
        sep = neg_index[0]
        slice_list = []
        size = 0
        for arg in args[:sep]:
            if arg is None:
                slice_list.append(size)
                size += 1
            else:
                slice_list.append(slice(size, size + arg))
                size += arg
        rslice_list = []
        rsize = 0
        for arg in args[:sep:-1]:
            if arg is None:
                rslice_list.append(-1-rsize)
                rsize += 1
            else:
                rslice_list.append(slice(None if not rsize else -rsize, -(rsize + arg)))
                rsize += arg
        slice_list.append(slice(size, rsize))
        slice_list.extend(reversed(rslice_list))
    def inner_wrapper(f):
        @functools.wraps(f)
        def wrapped_updater(keys, values):
            result = f(*[values[s] for s in slice_list])
            return (keys[:len(result)], result)
        return wrapped_updater
    return inner_wrapper