def is_cython_function(fn):
    """Checks if a function is compiled w/Cython."""
    if hasattr(fn, "__func__"):
        fn = fn.__func__  # Class method, static method
    name = type(fn).__name__
    return (
        name == "method_descriptor"
        or name == "cython_function_or_method"
        or name == "builtin_function_or_method"
    )