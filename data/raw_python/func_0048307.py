def _make_f_of_x(self, k_vec, e_vec, reaction):
        r"""
        Calculates :math:`F():math:` in eq. 12 (see Ale et al. 2013) for a specific reaction , :math:`k` and :math:`e`

        :param k_vec: the vector :math:`k`
        :param e_vec: the vector :math:`e`
        :param reaction: the equation of the reaction {:math:`a(x) in the model}
        :return: :math:`F()`
        """

        # product of all values of {x ^ (k - e)} for all combination of e and k
        prod = product([var ** (k_vec[i] - e_vec[i]) for i,var in enumerate(self.__species)])
        # multiply the product by the propensity {a(x)}
        return prod * reaction