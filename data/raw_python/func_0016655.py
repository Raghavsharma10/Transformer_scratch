def solvers(self, refresh=False, **filters):
        """Deprecated in favor of :meth:`.get_solvers`."""
        warnings.warn("'solvers' is deprecated in favor of 'get_solvers'.", DeprecationWarning)
        return self.get_solvers(refresh=refresh, **filters)