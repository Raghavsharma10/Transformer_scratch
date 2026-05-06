def dumpb(self, obj, *, max_nested_level=100):
        """Similar to ``dumps()``, but returns ``bytes`` instead of a ``string``"""
        self._max_nested_level = max_nested_level
        return self._encode(obj).encode('utf-8')