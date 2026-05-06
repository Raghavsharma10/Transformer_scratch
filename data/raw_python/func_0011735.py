def _rts_from_ra(ra, tspan, labels, block=True):
    """construct a RemoteTimeseries from a RemoteArray"""
    def _convert(a, tspan, labels):
        from nsim import Timeseries
        return Timeseries(a, tspan, labels)
    return distob.call(
            _convert, ra, tspan, labels, prefer_local=False, block=block)