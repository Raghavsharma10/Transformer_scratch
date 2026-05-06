def _pdf(self, phi):
        """
        Evaluate the _unnormalized_ flow PDF.

        """
        pdf = np.inner(self._vn, np.cos(np.outer(phi, self._n)))
        pdf *= 2.
        pdf += 1.

        return pdf