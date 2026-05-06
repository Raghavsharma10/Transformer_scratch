def _compute_closed_central_moments(self, central_from_raw_exprs, n_counter, k_counter):
        r"""
        Replace raw moment terms in central moment expressions by parameters (e.g. mean, variance, covariances)

        :param central_from_raw_exprs: the expression of central moments in terms of raw moments
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: the central moments where raw moments have been replaced by parametric expressions
        :rtype: `sympy.Matrix`
        """

        closed_central_moments = sp.Matrix([sp.Integer(self.__value)] * len(central_from_raw_exprs))
        return closed_central_moments