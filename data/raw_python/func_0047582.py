def close(self, mfk, central_from_raw_exprs, n_counter, k_counter):

        """
        In MFK, replaces symbol for high order (order == max_order+1) by parametric expressions.
        That is expressions depending on lower order moments such as means, variances, covariances and so on.

        :param mfk: the right hand side equations containing symbols for high order central moments
        :param central_from_raw_exprs: expressions of central moments in terms of raw moments
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: the modified MFK
        :rtype: `sympy.Matrix`
        """

        # we obtain expressions for central moments in terms of variances/covariances
        closed_central_moments = self._compute_closed_central_moments(central_from_raw_exprs, n_counter, k_counter)
        # set mixed central moment to zero iff univariate
        closed_central_moments = self._set_mixed_moments_to_zero(closed_central_moments, n_counter)

        # retrieve central moments from problem moment. Typically, :math: `[yx2, yx3, ...,yxN]`.

        # now we want to replace the new mfk (i.e. without highest order moment) any
        # symbol for highest order central moment by the corresponding expression (computed above)

        positive_n_counter = [n for n in n_counter if n.order > 0]
        substitutions_pairs = [(n.symbol, ccm) for n,ccm in
                               zip(positive_n_counter, closed_central_moments) if n.order > self.max_order]
        new_mfk = substitute_all(mfk, substitutions_pairs)

        return new_mfk