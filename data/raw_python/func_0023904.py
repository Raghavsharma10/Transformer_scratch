def _compute_dk_dtau_on_partition(self, tau, p):
        """Evaluate the term inside the sum of Faa di Bruno's formula for the given partition.
        
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
        y, r2l2 = self._compute_y(tau, return_r2l2=True)
        # Compute the d^(|pi|)f/dy term:
        dk_dtau = self._compute_dk_dy(y, len(p))
        # Multiply in each of the block terms:
        for b in p:
            dk_dtau *= self._compute_dy_dtau(tau, b, r2l2)
        return dk_dtau