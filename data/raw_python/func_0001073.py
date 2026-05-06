def flow(self, n, k, error=False, imaginary='nan'):
        r"""
        Calculate `v_n\{k\}`,
        the estimate of flow coefficient `v_n` from the `k`-particle cumulant.

        :param int n: Anisotropy order.
        :param int k: Correlation order.

        :param bool error:
            Whether to calculate statistical error (for `v_n\{2\}` only).
            If true, return a tuple ``(vn2, vn2_error)``.

        :param str imaginary: (optional)
            Determines behavior when the computed flow is imaginary:

            - ``'nan'`` (default) -- Return NaN and raise a ``RuntimeWarning``.
            - ``'negative'`` -- Return the negative absolute value.
            - ``'zero'`` -- Return ``0.0``.

        """
        cnk = self.cumulant(n, k, error=error)

        if error:
            cnk, cnk_err = cnk

        vnk_to_k = self._cnk_prefactor[k] * cnk
        kinv = 1/k

        if vnk_to_k >= 0:
            vnk = vnk_to_k**kinv
        else:
            if imaginary == 'negative':
                vnk = -1*(-vnk_to_k)**kinv
            elif imaginary == 'zero':
                vnk = 0.
            else:
                warnings.warn('Imaginary flow: returning NaN.', RuntimeWarning)
                vnk = float('nan')

        if k == 2 and error:
            return vnk, .5/np.sqrt(abs(cnk)) * cnk_err
        else:
            return vnk