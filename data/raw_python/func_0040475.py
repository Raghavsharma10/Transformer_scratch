def set_package(fxn):
    """Set __package__ on the returned module.
    This function is deprecated.
    """
    @functools.wraps(fxn)
    def set_package_wrapper(*args, **kwargs):
        warnings.warn('The import system now takes care of this automatically.',
                      DeprecationWarning, stacklevel=2)
        module = fxn(*args, **kwargs)
        if getattr(module, '__package__', None) is None:
            module.__package__ = module.__name__
            if not hasattr(module, '__path__'):
                module.__package__ = module.__package__.rpartition('.')[0]
        return module
    return set_package_wrapper