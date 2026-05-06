def _set_mixed_moments_to_zero(self, closed_central_moments, n_counter):
        r"""
        In univariate case, set the cross-terms to 0.

        :param closed_central_moments: matrix of closed central moment
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :return:  a matrix of new closed central moments with cross-terms equal to 0
        """

        positive_n_counter = [n for n in n_counter if n.order > 1]
        if self.is_multivariate:
            return closed_central_moments
        else:
            return [0 if n.is_mixed else ccm for n,ccm in zip(positive_n_counter, closed_central_moments)]