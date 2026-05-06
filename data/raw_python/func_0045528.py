def wrap(obj, wrapper=None, methods_to_add=(), name=None, skip=(), wrap_return_values=False, clear_cache=True):
    """
    Wrap module, class, function or another variable recursively

    :param Any obj: Object to wrap recursively
    :param Optional[Callable] wrapper: Wrapper to wrap functions and methods in (accepts function as argument)
    :param Collection[Callable] methods_to_add: Container of functions, which accept class as argument, and return \
    tuple of method name and method to add to all classes
    :param Optional[str] name: Name of module to wrap to (if `obj` is module)
    :param Collection[Union[str, type, Any]] skip: Items to skip wrapping (if an item of a collection is the str, wrap \
    will check the obj name, if an item of a collection is the type, wrap will check the obj type, else wrap will \
    check an item itself)
    :param bool wrap_return_values: If try, wrap return values of callables (only types, supported by wrap function \
    are supported)
    :param bool clear_cache: Clear wrapped objects cache after wrapping
    :return: Wrapped `obj`
    """
    result = _wrap(obj=obj,
                   wrapper=wrapper,
                   methods_to_add=methods_to_add,
                   name=name,
                   skip=skip,
                   wrap_return_values=wrap_return_values)
    if clear_cache:
        _wrapped_objs.clear()
    return result