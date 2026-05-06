def _dts_from_da(da, tspan, labels):
    """construct a DistTimeseries from a DistArray"""
    sublabels = labels[:]
    new_subarrays = []
    for i, ra in enumerate(da._subarrays):
        if isinstance(ra, RemoteTimeseries):
            new_subarrays.append(ra)
        else:
            if labels[da._distaxis]:
                sublabels[da._distaxis] = labels[da._distaxis][
                        da._si[i]:da._si[i+1]]
            new_subarrays.append(_rts_from_ra(ra, tspan, sublabels, False))
    new_subarrays = [distob.convert_result(ar) for ar in new_subarrays]
    da._subarrays = new_subarrays
    da.__class__ = DistTimeseries
    da.tspan = tspan
    da.labels = labels
    da.t = _Timeslice(da)
    return da