def __push_facvar_sparse(self, polynomial, block_index, row_offset, i, j):
        """Calculate the sparse vector representation of a polynomial
        and pushes it to the F structure.
        """
        width = self.block_struct[block_index - 1]
        # Preprocess the polynomial for uniform handling later
        # DO NOT EXPAND THE POLYNOMIAL HERE!!!!!!!!!!!!!!!!!!!
        # The simplify_polynomial bypasses the problem.
        # Simplifying here will trigger a bug in SymPy related to
        # the powers of daggered variables.
        # polynomial = polynomial.expand()
        if is_number_type(polynomial) or polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        # Identify its constituent monomials
        for element in elements:
            results = self._get_index_of_monomial(element)
            # k identifies the mapped value of a word (monomial) w
            for (k, coeff) in results:
                if k > -1 and coeff != 0:
                    self.F[row_offset + i * width + j, k] += coeff