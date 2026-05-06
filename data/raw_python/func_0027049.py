def is_classmethod(fn):
    """Returns whether f is a classmethod."""
    # This is True for bound methods
    if not inspect.ismethod(fn):
        return False
    if not hasattr(fn, "__self__"):
        return False
    im_self = fn.__self__
    # This is None for instance methods on classes, but True
    # for instance methods on instances.
    if im_self is None:
        return False
    # This is True for class methods of new- and old-style classes, respectively
    return isinstance(im_self, six.class_types)