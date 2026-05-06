def from_exception(cls, exception, retain_exc_info=True,
                       cause=None, find_cause=True):
        """Creates a failure object from a exception instance."""
        exc_info = (
            type(exception),
            exception,
            getattr(exception, '__traceback__', None)
        )
        return cls.from_exc_info(exc_info=exc_info,
                                 retain_exc_info=retain_exc_info,
                                 cause=cause, find_cause=find_cause)