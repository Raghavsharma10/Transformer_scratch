def _compute_dk_dtau(self, tau, n):
        r"""Evaluate :math:`dk/d\tau` at the specified locations with the specified derivatives.
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        n : :py:class:`Array`, (`D`,)
            Degree of derivative with respect to each dimension.
        
        Returns
        -------
            dk_dtau : :py:class:`Array`, (`M`,)
                Specified derivative at specified locations.
        """
        # Construct the derivative pattern:
        # For each dimension, this will contain the index of the dimension
        # repeated a number of times equal to the order of derivative with
        # respect to that dimension.
        # Example: For d^3 k(x, y, z) / dx^2 dy, n would be [2, 1, 0] and
        # deriv_pattern should be [0, 0, 1]. For k(x, y, z) deriv_pattern is [].
        deriv_pattern = []
        for idx in xrange(0, len(n)):
            deriv_pattern.extend(n[idx] * [idx])
        deriv_pattern = scipy.asarray(deriv_pattern, dtype=int)
        # Handle non-derivative case separately for efficiency:
        if len(deriv_pattern) == 0:
            return self._compute_k(tau)
        else:
            # Compute all partitions of the deriv_pattern:
            deriv_partitions = generate_set_partitions(deriv_pattern)
            # Compute the requested derivative using the multivariate Faa di Bruno's equation:
            dk_dtau = scipy.zeros(tau.shape[0])
            # Loop over the partitions:
            for partition in deriv_partitions:
                dk_dtau += self._compute_dk_dtau_on_partition(tau, partition)
            return dk_dtau