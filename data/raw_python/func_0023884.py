def _compute_k(self, tau):
        r"""Evaluate the kernel directly at the given values of `tau`.
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        
        Returns
        -------
        k : :py:class:`Array`, (`M`,)
            :math:`k(\tau)` (less the :math:`\sigma^2` prefactor).
        """
        y, r2l2 = self._compute_y(tau, return_r2l2=True)
        k = 2.0**(1.0 - self.nu) / scipy.special.gamma(self.nu) * y**(self.nu / 2.0) * scipy.special.kv(self.nu, scipy.sqrt(y))
        k[r2l2 == 0] = 1.0
        return k