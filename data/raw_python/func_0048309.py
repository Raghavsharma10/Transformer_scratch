def _make_s_pow_e(self, reac_idx, e_vec):
        """
        Compute s^e in equation 11  (see Ale et al. 2013)

        :param reac_idx: the index (that is the column in the stoichiometry matrix)
         of the reaction to consider.
        :type reac_idx: `int`
        :param e_vec: the vector e
        :return: a scalar (s^e)
        """
        return product([self.__stoichoimetry_matrix[i, reac_idx] ** e for i,e in enumerate(e_vec)])