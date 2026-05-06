def get(self, k_vec, e_counter):
        r"""
        Provides the terms needed for equation 11 (see Ale et al. 2013).
        This gives the expressions for :math:`\frac{d\beta}{dt}` in equation 9, these are the
        time dependencies of the mixed moments

        :param k_vec: :math:`k` in eq. 11
        :param e_counter: :math:`e` in eq. 11
        :return: :math:`\frac{d\beta}{dt}`
        """

        if len(e_counter) == 0:
            return sp.Matrix(1, len(self.__n_counter), lambda i, j: 0)

        # compute F(x) for EACH REACTION and EACH entry in the EKCOUNTER (eq. 12)
        f_of_x_vec = [self._make_f_of_x(k_vec, ek.n_vector, reac) for (reac, ek)
                      in itertools.product(self.__propensities, e_counter)]

        # compute <F> from f(x) (eq. 12). The result is a list in which each element is a
        # vector in which each element relates to an entry of counter
        f_expectation_vec = [self._make_f_expectation(f) for f in f_of_x_vec]

        # compute s^e for EACH REACTION and EACH entry in the EKCOUNTER . this is a list of scalars
        s_pow_e_vec = sp.Matrix([self._make_s_pow_e(reac_idx, ek.n_vector) for (reac_idx, ek)
                                 in itertools.product(range(len(self.__propensities)), e_counter)])

        # compute (k choose e) for EACH REACTION and EACH entry in the EKCOUNTER . This is a list of scalars.
        # Note that this does not depend on the reaction, so we can just repeat the result for each reaction
        k_choose_e_vec = sp.Matrix(
                [make_k_chose_e(ek.n_vector, k_vec) for ek in e_counter] *
                len(self.__propensities)
                )

        # compute the element-wise product of the three entities
        s_times_ke = s_pow_e_vec.multiply_elementwise(k_choose_e_vec)
        prod = [list(f * s_ke) for (f, s_ke) in zip(f_expectation_vec, s_times_ke)]

        # we have a list of vectors and we want to obtain a list of sums of all nth element together.
        # To do that we put all the data into a matrix in which each row is a different vector
        to_sum = sp.Matrix(prod)

        # then we sum over the columns -> row vector
        mixed_moments = sum_of_cols(to_sum)

        return mixed_moments