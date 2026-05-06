def ref(self):
        """ A reference (stored internally via a weakref) to an object
        that the backend system can use to obtain the low-level
        information of the "reference context". In Vispy this will
        typically be the CanvasBackend object.
        """
        # Clean
        self._refs = [r for r in self._refs if (r() is not None)]
        # Get ref
        ref = self._refs[0]() if self._refs else None
        if ref is not None:
            return ref
        else:
            raise RuntimeError('No reference for available for GLShared')