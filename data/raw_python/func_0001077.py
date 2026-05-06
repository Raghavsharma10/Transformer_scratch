def sample(self, multiplicity):
        r"""
        Randomly sample azimuthal angles `\phi`.

        :param int multiplicity: Number to sample.

        :returns: Array of sampled angles.

        """
        if self._n is None:
            return self._uniform_phi(multiplicity)

        # Since the flow PDF does not have an analytic inverse CDF, I use a
        # simple accept-reject sampling algorithm.  This is reasonably
        # efficient since for normal-sized vn, the PDF is close to flat.  Now
        # due to the overhead of Python functions, it's desirable to minimize
        # the number of calls to the random number generator.  Therefore I
        # sample numbers in chunks; most of the time only one or two chunks
        # should be needed.  Eventually, I might rewrite this with Cython, but
        # it's fast enough for now.

        N = 0  # number of phi that have been sampled
        phi = np.empty(multiplicity)  # allocate array for phi
        pdf_max = 1 + 2*self._vn.sum()  # sampling efficiency ~ 1/pdf_max

        while N < multiplicity:
            n_remaining = multiplicity - N
            n_to_sample = int(1.03*pdf_max*n_remaining)
            phi_chunk = self._uniform_phi(n_to_sample)
            phi_chunk = phi_chunk[self._pdf(phi_chunk) >
                                  np.random.uniform(0, pdf_max, n_to_sample)]
            K = min(phi_chunk.size, n_remaining)  # number of phi to take
            phi[N:N+K] = phi_chunk[:K]
            N += K

        return phi