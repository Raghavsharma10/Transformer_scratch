def _get_not_annotated(func, annotations=None):
    """Return non-optional parameters that are not annotated."""
    argspec = inspect.getfullargspec(func)
    args = argspec.args
    if argspec.defaults is not None:
        args = args[:-len(argspec.defaults)]
    if inspect.isclass(func) or inspect.ismethod(func):
        args = args[1:]  # Strip off ``cls`` or ``self``.
    kwonlyargs = argspec.kwonlyargs
    if argspec.kwonlydefaults is not None:
        kwonlyargs = kwonlyargs[:-len(argspec.kwonlydefaults)]
    annotations = annotations or argspec.annotations
    return [arg for arg in args + kwonlyargs if arg not in annotations]