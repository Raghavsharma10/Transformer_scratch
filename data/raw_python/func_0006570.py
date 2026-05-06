def protected_operation(fn):
    """
    Use this decorator to prevent an operation from being executed
    when the related uri resource is still in use.
    The parent_object must contain:
        * a request
            * with a registry.queryUtility(IReferencer)
    :raises pyramid.httpexceptions.HTTPConflict: Signals that we don't want to
        delete a certain URI because it's still in use somewhere else.
    :raises pyramid.httpexceptions.HTTPInternalServerError: Raised when we were
        unable to check that the URI is no longer being used.
    """
    @functools.wraps(fn)
    def advice(parent_object, *args, **kw):
        response = _advice(parent_object.request)
        if response is not None:
            return response
        else:
            return fn(parent_object, *args, **kw)

    return advice