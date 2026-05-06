def dumps(self, obj, *, max_nested_level=100):
        """Returns a string representing a JSON-encoding of ``obj``.

           The second optional ``max_nested_level`` argument controls the maximum
           allowed recursion/nesting level.

           See class description for details.
        """
        self._max_nested_level = max_nested_level
        return self._encode(obj)