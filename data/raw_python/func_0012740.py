def wrapped_help_text(wrapped_func):
    """Decorator to pass through the documentation from a wrapped function.
    """
    def decorator(wrapper_func):
        """The decorator.

        Parameters
        ----------
        f : callable
            The wrapped function.

        """
        wrapper_func.__doc__ = ('This method wraps the following method:\n\n' +
                                pydoc.text.document(wrapped_func))
        return wrapper_func
    return decorator