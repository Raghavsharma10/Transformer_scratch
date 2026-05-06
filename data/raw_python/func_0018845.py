def _get_tolerance(values):
    """Return some "numerical accuracy" to be expected for the
    given floating point value(s) (see method |trim|)."""
    tolerance = numpy.abs(values*1e-15)
    if hasattr(tolerance, '__setitem__'):
        tolerance[numpy.isinf(tolerance)] = 0.
    elif numpy.isinf(tolerance):
        tolerance = 0.
    return tolerance