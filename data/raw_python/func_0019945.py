def to_uncertain_func(x):
    """
    Transforms x into an UncertainFunction-compatible object,
    unless it is already an UncertainFunction (in which case x is returned 
    unchanged).

    Raises an exception unless 'x' belongs to some specific classes of
    objects that are known not to depend on UncertainFunction objects
    (which then cannot be considered as constants).
    """
    if isinstance(x, UncertainFunction):
        return x

    # ! In Python 2.6+, numbers.Number could be used instead, here:
    elif isinstance(x, CONSTANT_TYPES):
        # No variable => no derivative to define:
        return UncertainFunction([x] * npts)

    raise NotUpcast("%s cannot be converted to a number with" " uncertainty" % type(x))