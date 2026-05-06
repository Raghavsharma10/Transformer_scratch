def _process_monomial(self, monomial, n_vars):
        """Process a single monomial when building the moment matrix.
        """
        processed_monomial, coeff = separate_scalar_factor(monomial)
        # Are we substituting this moment?
        try:
            substitute = self.moment_substitutions[processed_monomial]
            if not isinstance(substitute, (int, float, complex)):
                result = []
                if not isinstance(substitute, Add):
                    args = [substitute]
                else:
                    args = substitute.args
                for arg in args:
                    if is_number_type(arg):
                        if iscomplex(arg):
                            result.append((0, coeff*complex(arg)))
                        else:
                            result.append((0, coeff*float(arg)))
                    else:
                        result += [(k, coeff*c2)
                                   for k, c2 in self._process_monomial(arg,
                                                                       n_vars)]
            else:
                result = [(0, coeff*substitute)]
        except KeyError:
            # Have we seen this monomial before?
            try:
                # If yes, then we improve sparsity by reusing the
                # previous variable to denote this entry in the matrix
                k = self.monomial_index[processed_monomial]
            except KeyError:
                # If no, it still may be possible that we have already seen its
                # conjugate. If the problem is real-valued, a monomial and its
                # conjugate should be equal (Hermiticity becomes symmetry)
                if not self.complex_matrix:
                    try:
                    # If we have seen the conjugate before, we just use the
                    # conjugate monomial instead
                        processed_monomial_adjoint = \
                              apply_substitutions(processed_monomial.adjoint(),
                                                  self.substitutions)
                        k = self.monomial_index[processed_monomial_adjoint]
                    except KeyError:
                        # Otherwise we define a new entry in the associated
                        # array recording the monomials, and add an entry in
                        # the moment matrix
                        k = n_vars + 1
                        self.monomial_index[processed_monomial] = k
                else:
                    k = n_vars + 1
                    self.monomial_index[processed_monomial] = k
            result = [(k, coeff)]
        return result