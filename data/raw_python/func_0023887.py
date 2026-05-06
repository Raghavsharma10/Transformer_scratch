def _compute_dk_dy(self, y, n):
        r"""Evaluate the derivative of the outer form of the Matern kernel.
        
        Uses the general Leibniz rule to compute the n-th derivative of:
        
        .. math::
        
            f(y) = \frac{2^{1-\nu}}{\Gamma(\nu)} y^{\nu/2} K_\nu(y^{1/2})
        
        Parameters
        ----------
        y : :py:class:`Array`, (`M`,)
            `M` inputs to evaluate at.
        n : non-negative scalar int.
            Order of derivative to compute.
        
        Returns
        -------
        dk_dy : :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        return 2.0**(1 - self.nu) / (scipy.special.gamma(self.nu)) * yn2Kn2Der(self.nu, y, n=n)