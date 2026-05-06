def protected_operation_with_request(fn):
    """
    Use this decorator to prevent an operation from being executed
    when the related uri resource is still in use.
    The request must contain a registry.queryUtility(IReferencer)
    :raises pyramid.httpexceptions.HTTPConflict: Signals that we don't want to
        delete a certain URI because it's still in use somewhere else.
    :raises pyramid.httpexceptions.HTTPInternalServerError: Raised when we were
        unable to check that the URI is no longer being used.
    """

    @functools.wraps(fn)
    def wrapped(request, *args, **kwargs):
        response = _advice(request)
        if response is not None:
            return response
        else:
            return fn(request, *args, **kwargs)

    return wrapped