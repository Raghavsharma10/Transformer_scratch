def decorate(decorator_cls, *args, **kwargs):
    """Creates a decorator function that applies the decorator_cls that was passed in."""
    global _wrappers

    wrapper_cls = _wrappers.get(decorator_cls, None)
    if wrapper_cls is None:

        class PythonWrapper(decorator_cls):
            pass

        wrapper_cls = PythonWrapper
        wrapper_cls.__name__ = decorator_cls.__name__ + "PythonWrapper"
        _wrappers[decorator_cls] = wrapper_cls

    def decorator(fn):
        wrapped = wrapper_cls(fn, *args, **kwargs)
        _update_wrapper(wrapped, fn)
        return wrapped

    return decorator