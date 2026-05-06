def copy(self, deep=False):
        """Copies this object (shallow or deep).

        :param deep: boolean indicating whether to do a deep copy (or a
                     shallow copy).
        """
        cause = self._cause
        if cause is not None:
            cause = cause.copy(deep=deep)
        exc_info = utils.copy_exc_info(self.exc_info, deep=deep)
        exc_args = self.exception_args
        exc_kwargs = self.exception_kwargs
        if deep:
            exc_args = copy.deepcopy(exc_args)
            exc_kwargs = copy.deepcopy(exc_kwargs)
        else:
            exc_args = tuple(exc_args)
            exc_kwargs = exc_kwargs.copy()
        # These are just simple int/strings, so deep copy doesn't really
        # matter/apply here (as they are immutable anyway).
        exc_type_names = tuple(self._exc_type_names)
        generated_on = self._generated_on
        if generated_on:
            generated_on = tuple(generated_on)
        # NOTE(harlowja): use `self.__class__` here so that we can work
        # with subclasses (assuming anyone makes one).
        return self.__class__(exc_info=exc_info,
                              exception_str=self.exception_str,
                              traceback_str=self.traceback_str,
                              exc_args=exc_args,
                              exc_kwargs=exc_kwargs,
                              exc_type_names=exc_type_names,
                              cause=cause, generated_on=generated_on)