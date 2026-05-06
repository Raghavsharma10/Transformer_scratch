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
        y = self._compute_y(tau)
        return y**(-self.params[1])