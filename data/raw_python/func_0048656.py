def pre_dispatch(self):
        """
        List of pre-dispatch methods from registered middleware.
        """
        middleware = sort_by_priority(self)
        return tuple(m.pre_dispatch for m in middleware if hasattr(m, 'pre_dispatch'))