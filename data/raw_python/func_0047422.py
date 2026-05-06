def _substitute_raw_with_central(self, central_moments_exprs, central_from_raw_exprs, n_counter, k_counter):
        r"""
        Takes the expressions for central moments, and substitute the symbols representing raw moments,
        by equivalent expressions in terms of central moment

        :param central_moments_exprs: a matrix of expressions for central moments.
        :param central_from_raw_exprs: central moment expressed in terms of raw moments
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]

        :return: expressions for central moments without raw moment
        """
        positiv_raw_moms_symbs = [raw.symbol for raw in k_counter if raw.order > 1]
        # The symbols for the corresponding central moment
        central_symbols= [central.symbol for central in n_counter if central.order > 1]
        # Now we state (central_symbols == central_from_raw_exprs)
        eq_to_solve = [cfr - cs for (cs, cfr) in zip(central_symbols, central_from_raw_exprs)]

        # And we solve this for the symbol of the corresponding raw moment. This gives an expression
        # of the symbol for raw moment in terms of central moments and lower order raw moment
        solved_xs = sp.Matrix([quick_solve(eq,raw) for (eq, raw) in zip(eq_to_solve, positiv_raw_moms_symbs)])

        # now we want to express raw moments only in terms od central moments and means
        # for instance if we have: :math:`x_1 = 1; x_2 = 2 +x_1 and  x_3 = x_2*x_1`, we should give:
        # :math:`x_1 = 1; x_2 = 2+1 and  x_3 = 1*(2+1)`
        # To achieve this, we recursively apply substitution as many times as the highest order (minus one)
        max_order = max([p.order for p in k_counter])

        for i in range(max_order - 1):
            substitution_pairs = zip(positiv_raw_moms_symbs, solved_xs)
            solved_xs = substitute_all(solved_xs, substitution_pairs)

        # we finally build substitution pairs to replace all raw moments
        substitution_pairs = zip(positiv_raw_moms_symbs, solved_xs)

        # apply this substitution to all elements of the central moment expressions matrix
        out_exprs = substitute_all(central_moments_exprs, substitution_pairs)

        return out_exprs