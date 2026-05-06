def sources(self):
        """
        Returns a dictionary of source methods found on this object,
        keyed on method name. Source methods are identified by
        (self, context) arguments on this object. For example:

        .. code-block:: python

            def f(self, context):
                    ...

        is a source method, but

        .. code-block:: python

            def f(self, ctx):
                ...

            is not.

        """

        try:
            return self._sources
        except AttributeError:
            self._sources = find_sources(self)

        return self._sources