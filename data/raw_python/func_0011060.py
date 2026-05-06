def _calculate_block_structure(self, inequalities, equalities,
                                   momentinequalities, momentequalities,
                                   extramomentmatrix, removeequalities,
                                   block_struct=None):
        """Calculates the block_struct array for the output file.
        """
        if block_struct is None:
            if self.verbose > 0:
                print("Calculating block structure...")
            self.block_struct = []
            if self.parameters is not None:
                self.block_struct += [1 for _ in self.parameters]
            for monomials in self.monomial_sets:
                if len(monomials) > 0 and isinstance(monomials[0], list):
                    self.block_struct.append(len(monomials[0]))
                else:
                    self.block_struct.append(len(monomials))
            if extramomentmatrix is not None:
                for _ in extramomentmatrix:
                    for monomials in self.monomial_sets:
                        if len(monomials) > 0 and \
                                isinstance(monomials[0], list):
                            self.block_struct.append(len(monomials[0]))
                        else:
                            self.block_struct.append(len(monomials))
        else:
            self.block_struct = block_struct
        degree_warning = False
        if inequalities is not None:
            self._n_inequalities = len(inequalities)
            n_tmp_inequalities = len(inequalities)
        else:
            self._n_inequalities = 0
            n_tmp_inequalities = 0
        constraints = flatten([inequalities])
        if momentinequalities is not None:
            self._n_inequalities += len(momentinequalities)
            constraints += momentinequalities
        if not removeequalities:
            constraints += flatten([equalities])
        monomial_sets = []
        for k, constraint in enumerate(constraints):
            # Find the order of the localizing matrix
            if k < n_tmp_inequalities or k >= self._n_inequalities:
                if isinstance(constraint, str):
                    ineq_order = 2 * self.level
                else:
                    if constraint.is_Relational:
                        constraint = convert_relational(constraint)
                    ineq_order = ncdegree(constraint)
                    if iscomplex(constraint):
                        self.complex_matrix = True
                if ineq_order > 2 * self.level:
                    degree_warning = True
                localization_order = (2*self.level - ineq_order)//2
                if self.level == -1:
                    localization_order = 0
                if self.localizing_monomial_sets is not None and \
                        self.localizing_monomial_sets[k] is not None:
                    localizing_monomials = self.localizing_monomial_sets[k]
                else:
                    index = find_variable_set(self.variables, constraint)
                    localizing_monomials = \
                        pick_monomials_up_to_degree(self.monomial_sets[index],
                                                    localization_order)
                ln = len(localizing_monomials)
                if ln == 0:
                    localizing_monomials = [S.One]
            else:
                localizing_monomials = [S.One]
                ln = 1
            localizing_monomials = unique(localizing_monomials)
            monomial_sets.append(localizing_monomials)
            if k < self._n_inequalities:
                self.block_struct.append(ln)
            else:
                monomial_sets += [None for _ in range(ln*(ln+1)//2-1)]
                monomial_sets.append(localizing_monomials)
                monomial_sets += [None for _ in range(ln*(ln+1)//2-1)]
                self.block_struct += [1 for _ in range(ln*(ln+1))]

        if degree_warning and self.verbose > 0:
            print("A constraint has degree %d. Either choose a higher level "
                  "relaxation or ensure that a mixed-order relaxation has the"
                  " necessary monomials" % (ineq_order), file=sys.stderr)

        if momentequalities is not None:
            for moment_eq in momentequalities:
                self._moment_equalities.append(moment_eq)
                if not removeequalities:
                    monomial_sets += [[S.One], [S.One]]
                    self.block_struct += [1, 1]
        self.localizing_monomial_sets = monomial_sets