def sinks(self):
        """
        Returns a dictionary of sink methods found on this object,
        keyed on method name. Sink methods are identified by
        (self, context) arguments on this object. For example:

        def f(self, context):
            ...

        is a sink method, but

        def f(self, ctx):
            ...

        is not.

        """

        try:
            return self._sinks
        except AttributeError:
            self._sinks = find_sinks(self)

        return self._sinks