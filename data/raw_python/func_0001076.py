def pdf(self, phi):
        r"""
        Evaluate the flow PDF `dN/d\phi`.

        :param array-like phi: Azimuthal angles.

        :returns: The flow PDF evaluated at ``phi``.

        """
        if self._n is None:
            pdf = np.empty_like(phi)
            pdf.fill(.5/np.pi)
            return pdf

        phi = np.asarray(phi)

        pdf = self._pdf(phi)
        pdf /= 2.*np.pi

        return pdf