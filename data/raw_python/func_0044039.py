def translate(self, vector):
        """Translates `Atom`.

        Parameters
        ----------
        vector : 3D Vector (tuple, list, numpy.array)
            Vector used for translation.
        inc_alt_states : bool, optional
            If true, will rotate atoms in all states i.e. includes
            alternate conformations for sidechains.
        """
        vector = numpy.array(vector)
        self._vector += numpy.array(vector)
        return