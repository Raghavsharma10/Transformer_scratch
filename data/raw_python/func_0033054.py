def filtered(f):
    '''
    Decorator function that wraps functions returning pandas
    dataframes, such that the dataframe is filtered
    according to left and right bounds set.
    '''

    def _filter(f, self, *args, **kwargs):
        frame = f(self, *args, **kwargs)
        ret = type(self)(frame)
        ret._lbound = self._lbound
        ret._rbound = self._rbound
        return ret

    if HAS_DECORATOR:
        return decorator(_filter, f)
    else:
        def err_func(*args, **kwargs):
            raise RuntimeError("`pip install decorator` required")
        return err_func