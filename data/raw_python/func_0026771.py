def decorate_callable(self, func):
        """
        Decorates a function with the db_mutex decorator by using this class as a context manager around
        it.
        """
        def wrapper(*args, **kwargs):
            try:
                with self:
                    result = func(*args, **kwargs)
                return result
            except DBMutexError as e:
                if self.suppress_acquisition_exceptions:
                    LOG.error(e)
                else:
                    raise e
        functools.update_wrapper(wrapper, func)
        return wrapper