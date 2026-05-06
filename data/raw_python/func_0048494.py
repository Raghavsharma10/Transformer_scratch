def add_param(param):
    # type: (Param) -> Callable
    """
    Add parameter, you should probably use on of :meth:`path_param`, :meth:`query_param`,
    :meth:`body_param`, or :meth:`header_param`.
    """
    def inner(o):
        try:
            getattr(o, 'parameters').add(param)
        except AttributeError:
            setattr(o, 'parameters', {param})
        return o
    return inner