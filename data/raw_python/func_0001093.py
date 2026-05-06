def ecc(self, n):
        r"""
        Calculate eccentricity harmonic `\varepsilon_n`.

        :param int n: Eccentricity order.

        """
        ny, nx = self._profile.shape
        xmax, ymax = self._xymax
        xcm, ycm = self._cm

        # create (X, Y) grids relative to CM
        Y, X = np.mgrid[ymax:-ymax:1j*ny, -xmax:xmax:1j*nx]
        X -= xcm
        Y -= ycm

        # create grid of weights = profile * R^n
        Rsq = X*X + Y*Y
        if n == 1:
            W = np.sqrt(Rsq, out=Rsq)
        elif n == 2:
            W = Rsq
        else:
            if n & 1:  # odd n
                W = np.sqrt(Rsq)
            else:  # even n
                W = np.copy(Rsq)
            # multiply by R^2 until W = R^n
            for _ in range(int((n-1)/2)):
                W *= Rsq
        W *= self._profile

        # create grid of e^{i*n*phi} * W
        i_n_phi = np.zeros_like(X, dtype=complex)
        np.arctan2(Y, X, out=i_n_phi.imag)
        i_n_phi.imag *= n
        exp_phi = np.exp(i_n_phi, out=i_n_phi)
        exp_phi *= W

        return abs(exp_phi.sum()) / W.sum()