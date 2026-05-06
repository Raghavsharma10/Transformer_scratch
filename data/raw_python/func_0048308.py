def _make_f_expectation(self, expr):
        """
        Calculates :math:`<F>` in eq. 12 (see Ale et al. 2013) to calculate :math:`<F>` for EACH VARIABLE combination.

        :param expr: an expression
        :return: a column vector. Each row correspond to an element of counter.
        :rtype: :class:`sympy.Matrix`
        """
        # compute derivatives for EACH ENTRY in COUNTER

        derives = sp.Matrix([derive_expr_from_counter_entry(expr, self.__species, tuple(c.n_vector))
                             for c in self.__n_counter])



        # Computes the factorial terms for EACH entry in COUNTER
        factorial_terms = sp.Matrix([get_one_over_n_factorial(tuple(c.n_vector)) for c in self.__n_counter])

        # Element wise product of the two vectors
        te_vector= derives.multiply_elementwise(factorial_terms)

        return te_vector