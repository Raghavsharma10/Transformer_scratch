def __process_equalities(self, equalities, momentequalities):
        """Generate localizing matrices

        Arguments:
        equalities -- list of equality constraints
        equalities -- list of moment equality constraints
        """
        monomial_sets = []
        n_rows = 0
        le = 0
        if equalities is not None:
            for equality in equalities:
                le += 1
                # Find the order of the localizing matrix
                if equality.is_Relational:
                    equality = convert_relational(equality)
                eq_order = ncdegree(equality)
                if eq_order > 2 * self.level:
                    raise Exception("An equality constraint has degree %d. "
                                    "Choose a higher level of relaxation."
                                    % eq_order)
                localization_order = (2 * self.level - eq_order)//2
                index = find_variable_set(self.variables, equality)
                localizing_monomials = \
                    pick_monomials_up_to_degree(self.monomial_sets[index],
                                                localization_order)
                if len(localizing_monomials) == 0:
                    localizing_monomials = [S.One]
                localizing_monomials = unique(localizing_monomials)
                monomial_sets.append(localizing_monomials)
                n_rows += len(localizing_monomials) * \
                    (len(localizing_monomials) + 1) // 2
        if momentequalities is not None:
            for _ in momentequalities:
                le += 1
                monomial_sets.append([S.One])
                n_rows += 1
        A = np.zeros((n_rows, self.n_vars + 1), dtype=self.F.dtype)
        n_rows = 0
        if self._parallel:
            pool = Pool()
        for i, equality in enumerate(flatten([equalities, momentequalities])):
            func = partial(moment_of_entry, monomials=monomial_sets[i],
                           ineq=equality, substitutions=self.substitutions)
            lm = len(monomial_sets[i])
            if self._parallel and lm > 1:
                chunksize = max(int(np.sqrt(lm*lm/2) /
                                    cpu_count()), 1)
                iter_ = pool.map(func, ([row, column] for row in range(lm)
                                        for column in range(row, lm)),
                                 chunksize)
            else:
                iter_ = imap(func, ([row, column] for row in range(lm)
                                    for column in range(row, lm)))
            # Process M_y(gy)(u,w) entries
            for row, column, polynomial in iter_:
                # Calculate the moments of polynomial entries
                if isinstance(polynomial, str):
                    self.__parse_expression(equality, -1, A[n_rows])
                else:
                    A[n_rows] = self._get_facvar(polynomial)
                n_rows += 1
                if self.verbose > 0:
                    sys.stdout.write("\r\x1b[KProcessing %d/%d equalities..." %
                                     (i+1, le))
                    sys.stdout.flush()
        if self._parallel:
            pool.close()
            pool.join()

        if self.verbose > 0:
            sys.stdout.write("\n")
        return A