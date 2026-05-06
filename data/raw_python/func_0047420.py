def _generate_problem_left_hand_side(self, n_counter, k_counter):
        """
        Generate the left hand side of the ODEs. This is simply the
        symbols for the corresponding moments.
        Note that, in principle, they are in of course fact the
        time derivative of the moments.

        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: a list of the problem left hand sides
        :rtype: list[:class:`sympy.Symbol`]
        """

        # concatenate the symbols for first order raw moments (means)
        prob_moments_over_dt = [k for k in k_counter if k.order == 1]
        # and the higher order central moments (variances, covariances,...)
        prob_moments_over_dt += [n for n in n_counter if self.__max_order >= n.order > 1]


        return prob_moments_over_dt