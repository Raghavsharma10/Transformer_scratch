def _compute_closed_central_moments(self, central_from_raw_exprs, n_counter, k_counter):
        r"""
        Computes parametric expressions (e.g. in terms of mean, variance, covariances) for all central moments
        up to max_order + 1 order.

        :param central_from_raw_exprs: the expression of central moments in terms of raw moments
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: the central moments where raw moments have been replaced by parametric expressions
        :rtype: `sympy.Matrix`
        """

        closed_raw_moments = self._compute_raw_moments(n_counter, k_counter)
        assert(len(central_from_raw_exprs) == len(closed_raw_moments))
        # raw moment lef hand side symbol
        raw_symbols = [raw.symbol for raw in k_counter if raw.order > 1]

        # we want to replace raw moments symbols with closed raw moment expressions (in terms of variances/means)
        substitution_pairs = zip(raw_symbols, closed_raw_moments)
        # so we can obtain expression of central moments in terms of low order raw moments
        closed_central_moments = substitute_all(central_from_raw_exprs, substitution_pairs)
        return closed_central_moments