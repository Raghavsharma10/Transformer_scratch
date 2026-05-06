def create_shared(self, name, ref):
        """ For the app backends to create the GLShared object.

        Parameters
        ----------
        name : str
            The name.
        ref : object
            The reference.
        """
        if self._shared is not None:
            raise RuntimeError('Can only set_shared once.')
        self._shared = GLShared(name, ref)