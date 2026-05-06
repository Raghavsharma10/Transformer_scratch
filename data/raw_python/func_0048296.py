def _compute_closed_central_moments(self, central_from_raw_exprs, n_counter, k_counter):
        """
        Computes parametric expressions (e.g. in terms of mean, variance, covariances) for all central moments
        up to max_order + 1 order.

        :param central_from_raw_exprs:
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: a vector of parametric expression for central moments
        """
        n_species = len([None for pm in k_counter if pm.order == 1])
        covariance_matrix = sp.Matrix(n_species, n_species, lambda x,y: self._get_covariance_symbol(n_counter,x,y))
        positive_n_counter = [n for n in n_counter if n.order > 1]
        out_mat = [self._compute_one_closed_central_moment(n, covariance_matrix) for n in positive_n_counter ]
        return sp.Matrix(out_mat)