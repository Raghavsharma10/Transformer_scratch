def _wrap_set_op(self, fun, arg):
        """Wrap built-in set operations for RangeSet to workaround built-in set
        base class issues (RangeSet.__new/init__ not called)"""
        result = fun(self, arg)
        result._autostep = self._autostep
        result.padding = self.padding
        return result