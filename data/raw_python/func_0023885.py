def _compute_y(self, tau, return_r2l2=False):
        r"""Covert tau to :math:`y=2\nu\sum_i(\tau_i^2/l_i^2)`.
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        return_r2l2 : bool, optional
            Set to True to return a tuple of (`y`, `r2l2`). Default is False
            (only return `y`).
        
        Returns
        -------
        y : :py:class:`Array`, (`M`,)
            Inner argument of function.
        r2l2 : :py:class:`Array`, (`M`,)
            Anisotropically scaled distances. Only returned if `return_r2l2` is True.
        """
        r2l2 = self._compute_r2l2(tau)
        y = 2.0 * self.nu * r2l2
        if return_r2l2:
            return (y, r2l2)
        else:
            return y