def _compute_raw_moments(self, n_counter, k_counter):
        r"""
        Compute :math:`X_i`
        Gamma type 1: :math:`X_i = \frac {\beta_i}{\beta_0}Y_0 + Y_i`
        Gamma type 2: :math:`X_i = \sum_{k=0}^{i}  \frac {\beta_i}{\beta_k}Y_k`

        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: a vector of parametric expression for raw moments
        """

        alpha_multipliers, beta_multipliers = self._get_parameter_symbols(n_counter, k_counter)

        out_mat = sp.Matrix([a * b for a,b in zip(alpha_multipliers, beta_multipliers)])
        out_mat = out_mat.applyfunc(sp.expand)
        return out_mat