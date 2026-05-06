def lazy_constant(fn):
    """Decorator to make a function that takes no arguments use the LazyConstant class."""

    class NewLazyConstant(LazyConstant):
        @functools.wraps(fn)
        def __call__(self):
            return self.get_value()

    return NewLazyConstant(fn)