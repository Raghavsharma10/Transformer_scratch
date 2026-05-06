def is_cython_or_generator(fn):
    """Returns whether this function is either a generator function or a Cythonized function."""
    if hasattr(fn, "__func__"):
        fn = fn.__func__  # Class method, static method
    if inspect.isgeneratorfunction(fn):
        return True
    name = type(fn).__name__
    return (
        name == "generator"
        or name == "method_descriptor"
        or name == "cython_function_or_method"
        or name == "builtin_function_or_method"
    )