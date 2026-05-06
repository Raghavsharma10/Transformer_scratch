def _ufunc_wrap(out_arr, ufunc, method, i, inputs, **kwargs):
    """After using the superclass __numpy_ufunc__ to route ufunc computations 
    on the array data, convert any resulting ndarray, RemoteArray and DistArray
    instances into Timeseries, RemoteTimeseries and DistTimeseries instances
    if appropriate"""
    # Assigns tspan/labels to an axis only if inputs do not disagree on them.
    shape = out_arr.shape
    ndim = out_arr.ndim
    if ndim is 0 or shape[0] is 0:
        # not a timeseries
        return out_arr
    candidates = [a.tspan for a in inputs if (hasattr(a, 'tspan') and
                                              a.shape[0] == shape[0])]
    # Expensive to validate all tspans are the same. check start and end t
    starts = [tspan[0] for tspan in candidates]
    ends = [tspan[-1] for tspan in candidates]
    if len(set(starts)) != 1 or len(set(ends)) != 1:
        # inputs cannot agree on tspan
        return out_arr
    else:
        new_tspan = candidates[0]
    new_labels = [None]
    for i in range(1, ndim):
        candidates = [a.labels[i] for a in inputs if (hasattr(a, 'labels') and 
                 a.shape[i] == shape[i] and a.labels[i] is not None)] 
        if len(candidates) is 1:
            new_labels.append(candidates[0])
        elif (len(candidates) > 1 and all(labs[j] == candidates[0][j] for 
                labs in candidates[1:] for j in range(shape[i]))):
            new_labels.append(candidates[0])
        else:
            new_labels.append(None)
    if isinstance(out_arr, np.ndarray):
        return Timeseries(out_arr, new_tspan, new_labels)
    elif isinstance(out_arr, distob.RemoteArray):
        return _rts_from_ra(out_arr, new_tspan, new_labels)
    elif (isinstance(out_arr, distob.DistArray) and
          all(isinstance(ra, RemoteTimeseries) for ra in out_arr._subarrays)):
        return _dts_from_da(out_arr, new_tspan, new_labels)
    else:
        return out_arr