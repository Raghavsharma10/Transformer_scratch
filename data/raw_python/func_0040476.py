def set_loader(fxn):
    """Set __loader__ on the returned module.
    This function is deprecated.
    """
    @functools.wraps(fxn)
    def set_loader_wrapper(self, *args, **kwargs):
        warnings.warn('The import system now takes care of this automatically.',
                      DeprecationWarning, stacklevel=2)
        module = fxn(self, *args, **kwargs)
        if getattr(module, '__loader__', None) is None:
            module.__loader__ = self
        return module
    return set_loader_wrapper