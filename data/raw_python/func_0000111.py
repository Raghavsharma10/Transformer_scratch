def operations():
    """
    Class decorator stores all calls into list.
    Can be used until .invalidate() is called.
    :return: decorated class
    """
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):

            self = args[0]
            assert self.__can_use, "User operation queue only in 'with' block"

            def defaults_dict():
                f_args, varargs, keywords, defaults = inspect.getargspec(func)
                defaults = defaults or []
                return dict(zip(f_args[-len(defaults)+len(args[1:]):], defaults[len(args[1:]):]))

            route_args = dict(defaults_dict().items() + kwargs.items())

            func(*args, **kwargs)
            self.operations.append((func.__name__, args[1:], route_args, ))

        return wrapped_func
    def decorate(clazz):

        for attr in clazz.__dict__:
            if callable(getattr(clazz, attr)):
                setattr(clazz, attr, decorator(getattr(clazz, attr)))
        def __init__(self):  # simple parameter-less constructor
            self.operations = []
            self.__can_use = True
        def invalidate(self):
            self.__can_use = False
        clazz.__init__ = __init__
        clazz.invalidate = invalidate
        return clazz
    return decorate