def _compute_dk_dtau_on_partition(self, tau, p):
        """Evaluate the term inside the sum of Faa di Bruno's formula for the given partition.
        
        Overrides the version from :py:class:`gptools.kernel.core.ChainRuleKernel`
        in order to get the correct behavior at the origin.
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        p : list of :py:class:`Array`
            Each element is a block of the partition representing the
            derivative orders to use.
        
        Returns
        -------
        dk_dtau : :py:class:`Array`, (`M`,)
            The specified derivatives over the given partition at the specified
            locations.
        """
        # Find the derivative order:
        n = len(p)
        y, r2l2 = self._compute_y(tau, return_r2l2=True)
        # Keep track of how many times a given variable has a block of length 1:
        n1 = 0
        # Build the dy/dtau factor up iteratively:
        dy_dtau_factor = scipy.ones_like(y)
        for b in p:
            # If the partial derivative is exactly zero there is no sense in
            # continuing the computation:
            if (len(b) > 2) or ((len(b) == 2) and (b[0] != b[1])):
                return scipy.zeros_like(y)
            dy_dtau_factor *= self._compute_dy_dtau(tau, b, r2l2)
            # Count the number of blocks of length 1:
            if len(b) == 1:
                n1 += 1.0
        # Compute d^(|pi|)f/dy^(|pi|) term:
        dk_dy = self._compute_dk_dy(y, n)
        if n1 > 0:
            mask = (y == 0.0)
            tau_pow = 2 * (self.nu - n) + n1
            if tau_pow == 0:
                # In this case the limit does not exist, so it is set to NaN:
                dk_dy[mask] = scipy.nan
            elif tau_pow > 0:
                dk_dy[mask] = 0.0
                
        return dk_dy * dy_dtau_factor