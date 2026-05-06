def add_ref(self, name, ref):
        """ Add a reference for the backend object that gives access
        to the low level context. Used in vispy.app.canvas.backends.
        The given name must match with that of previously added
        references.
        """
        if self._name is None:
            self._name = name
        elif name != self._name:
            raise RuntimeError('Contexts can only share between backends of '
                               'the same type')
        self._refs.append(weakref.ref(ref))