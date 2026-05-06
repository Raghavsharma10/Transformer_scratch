def correlation(self, n, k, error=False):
        r"""
        Calculate `\langle k \rangle_n`,
        the `k`-particle correlation function for `n`\ th-order anisotropy.

        :param int n: Anisotropy order.
        :param int k: Correlation order.

        :param bool error:
            Whether to calculate statistical error
            (for `\langle 2 \rangle` only).
            If true, return a tuple ``(corr, corr_error)``.

        """
        self._calculate_corr(n, k)
        corr_nk = self._corr[n][k]

        if error:
            self._calculate_corr_err(n, k)
            return corr_nk, self._corr_err[n][k]
        else:
            return corr_nk