def from_exc_info(cls, exc_info=None,
                      retain_exc_info=True,
                      cause=None, find_cause=True):
        """Creates a failure object from a ``sys.exc_info()`` tuple."""
        if exc_info is None:
            exc_info = sys.exc_info()
            if not any(exc_info):
                raise NoActiveException("No exception currently"
                                        " being handled")
        # This should always be the (type, value, traceback) tuple,
        # either from a prior sys.exc_info() call or from some other
        # creation...
        if len(exc_info) != 3:
            raise ValueError("Provided 'exc_info' must contain three"
                             " elements")
        exc_type, exc_val, exc_tb = exc_info
        try:
            if exc_type is None or exc_val is None:
                raise ValueError("Invalid exception tuple (exception"
                                 " type and exception value must"
                                 " be provided)")
            exc_args = tuple(getattr(exc_val, 'args', []))
            exc_kwargs = dict(getattr(exc_val, 'kwargs', {}))
            exc_type_names = utils.extract_roots(exc_type)
            if not exc_type_names:
                exc_type_name = reflection.get_class_name(
                    exc_val, truncate_builtins=False)
                # This should only be possible if the exception provided
                # was not really an exception...
                raise TypeError("Invalid exception type '%s' (not an"
                                " exception)" % (exc_type_name))
            exception_str = utils.exception_message(exc_val)
            if hasattr(exc_val, '__traceback_str__'):
                traceback_str = exc_val.__traceback_str__
            else:
                if exc_tb is not None:
                    traceback_str = '\n'.join(
                        traceback.format_exception(*exc_info))
                else:
                    traceback_str = ''
            if not retain_exc_info:
                exc_info = None
            if find_cause and cause is None:
                cause = cls._extract_cause(exc_val)
            return cls(exc_info=exc_info, exc_args=exc_args,
                       exc_kwargs=exc_kwargs, exception_str=exception_str,
                       exc_type_names=exc_type_names, cause=cause,
                       traceback_str=traceback_str,
                       generated_on=sys.version_info[0:2])
        finally:
            del exc_type, exc_val, exc_tb