def update_gl_state(self, *args, **kwargs):
        """Modify the set of GL state parameters to use when drawing

        Parameters
        ----------
        *args : tuple
            Arguments.
        **kwargs : dict
            Keyword argments.
        """
        for v in self._subvisuals:
            v.update_gl_state(*args, **kwargs)