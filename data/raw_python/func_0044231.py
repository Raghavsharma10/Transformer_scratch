def compile_dependencies(self, sourcepath, include_self=True):
        """
        Same as inherit method but the default value for keyword argument
        ``ìnclude_self`` is ``True``.
        """
        return super(SassProjectEventHandler, self).compile_dependencies(
            sourcepath,
            include_self=include_self
        )