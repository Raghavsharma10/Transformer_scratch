def _process_monomial(self, monomial, n_vars):
        """Process a single monomial when building the moment matrix.
        """
        coeff, monomial = monomial.as_coeff_Mul()
        k = 0
        # Have we seen this monomial before?
        conjugate = False
        try:
            # If yes, then we improve sparsity by reusing the
            # previous variable to denote this entry in the matrix
            k = self.monomial_index[monomial]
        except KeyError:
            # An extra round of substitutions is granted on the conjugate of
            # the monomial if all the variables are Hermitian
            daggered_monomial = \
                apply_substitutions(Dagger(monomial), self.substitutions,
                                    self.pure_substitution_rules)
            try:
                k = self.monomial_index[daggered_monomial]
                conjugate = True
            except KeyError:
                # Otherwise we define a new entry in the associated
                # array recording the monomials, and add an entry in
                # the moment matrix
                k = n_vars + 1
                self.monomial_index[monomial] = k
        if conjugate:
            k = -k
        return k, coeff