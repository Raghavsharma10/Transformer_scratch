def construct_exc_class(cls):
    """Constructs proxy class for the exception."""

    class ProxyException(cls, BaseException):
        __pep3134__ = True

        @property
        def __traceback__(self):
            if self.__fixed_traceback__:
                return self.__fixed_traceback__

            current_exc, current_tb = sys.exc_info()[1:]
            if current_exc is self:
                return current_tb

        def __init__(self, instance=None):  # pylint: disable=W0231
            self.__original_exception__ = instance
            self.__fixed_traceback__ = None

        def __getattr__(self, item):
            return getattr(self.__original_exception__, item)

        def __repr__(self):
            return repr(self.__original_exception__)

        def __str__(self):
            return str(self.__original_exception__)

        def with_traceback(self, traceback):
            instance = copy.copy(self)
            instance.__fixed_traceback__ = traceback
            return instance

    ProxyException.__name__ = cls.__name__

    return ProxyException