def decorate_method(wrapped):
    """The decorated method will return a |Lines| object including
    a method header.  However, the |Lines| object will be empty if
    the respective model does not implement a method with the same
    name as the wrapped method.
    """
    def wrapper(self):
        lines = Lines()
        if hasattr(self.model, wrapped.__name__):
            print('            . %s' % wrapped.__name__)
            lines.add(1, method_header(wrapped.__name__, nogil=True))
            for line in wrapped(self):
                lines.add(2, line)
        return lines
    functools.update_wrapper(wrapper, wrapped)
    wrapper.__doc__ = 'Lines of model method %s.' % wrapped.__name__
    return property(wrapper)