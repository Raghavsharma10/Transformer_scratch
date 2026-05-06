def _compute_dy_dtau(self, tau, b, r2l2):
        r"""Evaluate the derivative of the inner argument of the Matern kernel.
        
        Take the derivative of
        
        .. math::
        
            y = 2 \nu \sum_i(\tau_i^2 / l_i^2)
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        b : :py:class:`Array`, (`P`,)
            Block specifying derivatives to be evaluated.
        r2l2 : :py:class:`Array`, (`M`,)
            Precomputed anisotropically scaled distance.
        
        Returns
        -------
        dy_dtau: :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        if len(b) == 0:
            return self._compute_y(tau)
        elif len(b) == 1:
            return 4.0 * self.nu * tau[:, b[0]] / (self.params[2 + b[0]])**2.0
        elif (len(b) == 2) and (b[0] == b[1]):
            return 4.0 * self.nu / (self.params[2 + b[0]])**2.0
        else:
            return scipy.zeros_like(r2l2)