def update_gl_state(self, *args, **kwargs):
        """Modify the set of GL state parameters to use when drawing

        Parameters
        ----------
        *args : tuple
            Arguments.
        **kwargs : dict
            Keyword argments.
        """
        if len(args) == 1:
            self._vshare.gl_state['preset'] = args[0]
        elif len(args) != 0:
            raise TypeError("Only one positional argument allowed.")
        self._vshare.gl_state.update(kwargs)