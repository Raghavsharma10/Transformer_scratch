def cumulant(self, n, k, error=False):
        r"""
        Calculate `c_n\{k\}`,
        the `k`-particle cumulant for `n`\ th-order anisotropy.

        :param int n: Anisotropy order.
        :param int k: Correlation order.

        :param bool error:
            Whether to calculate statistical error (for `c_n\{2\}` only).
            If true, return a tuple ``(cn2, cn2_error)``.

        """
        corr_nk = self.correlation(n, k, error=error)

        if k == 2:
            return corr_nk
        elif k == 4:
            corr_n2 = self.correlation(n, 2)
            return corr_nk - 2*corr_n2*corr_n2