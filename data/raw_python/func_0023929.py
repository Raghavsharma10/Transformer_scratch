def _compute_dk_dy(self, y, n):
        """Evaluate the derivative of the outer form of the RQ kernel.
        
        Parameters
        ----------
        y : :py:class:`Array`, (`M`,)
            `M` inputs to evaluate at.
        n : non-negative scalar int
            Order of derivative to compute.
        
        Returns
        -------
        dk_dy : :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        p = fixed_poch(1.0 - self.params[1] - n, n)
        return p * y**(-self.params[1] - n)