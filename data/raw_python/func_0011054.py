def _get_index_of_monomial(self, element, enablesubstitution=True,
                               daggered=False):
        """Returns the index of a monomial.
        """
        result = []
        processed_element, coeff1 = separate_scalar_factor(element)
        if processed_element in self.moment_substitutions:
            r = self._get_index_of_monomial(self.moment_substitutions[processed_element], enablesubstitution)
            return [(k, coeff*coeff1) for k, coeff in r]
        if enablesubstitution:
            processed_element = \
                apply_substitutions(processed_element, self.substitutions,
                                    self.pure_substitution_rules)
        # Given the monomial, we need its mapping L_y(w) to push it into
        # a corresponding constraint matrix
        if is_number_type(processed_element):
            return [(0, coeff1)]
        elif processed_element.is_Add:
            monomials = processed_element.args
        else:
            monomials = [processed_element]
        for monomial in monomials:
            monomial, coeff2 = separate_scalar_factor(monomial)
            coeff = coeff1*coeff2
            if is_number_type(monomial):
                result.append((0, coeff))
                continue
            k = -1
            if monomial != 0:
                if monomial.as_coeff_Mul()[0] < 0:
                    monomial = -monomial
                    coeff = -1.0 * coeff
            try:
                new_element = self.moment_substitutions[monomial]
                r = self._get_index_of_monomial(self.moment_substitutions[new_element], enablesubstitution)
                result += [(k, coeff*coeff3) for k, coeff3 in r]
            except KeyError:
                try:
                    k = self.monomial_index[monomial]
                    result.append((k, coeff))
                except KeyError:
                    if not daggered:
                        dag_result = self._get_index_of_monomial(monomial.adjoint(),
                                                                 daggered=True)
                        result += [(k, coeff0*coeff) for k, coeff0 in dag_result]
                    else:
                        raise RuntimeError("The requested monomial " +
                                           str(monomial) + " could not be found.")
        return result