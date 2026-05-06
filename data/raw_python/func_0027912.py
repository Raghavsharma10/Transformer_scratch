def score_intersect(self, term1, term2, **kwargs):

        """
        Compute the geometric area of the overlap between the kernel density
        estimates of two terms.

        Args:
            term1 (str)
            term2 (str)

        Returns: float
        """

        t1_kde = self.kde(term1, **kwargs)
        t2_kde = self.kde(term2, **kwargs)

        # Integrate the overlap.
        overlap = np.minimum(t1_kde, t2_kde)
        return np.trapz(overlap)