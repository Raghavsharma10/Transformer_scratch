def process_constraints(self, inequalities=None, equalities=None,
                            momentinequalities=None, momentequalities=None,
                            block_index=0, removeequalities=False):
        """Process the constraints and generate localizing matrices. Useful
        only if the moment matrix already exists. Call it if you want to
        replace your constraints. The number of the respective types of
        constraints and the maximum degree of each constraint must remain the
        same.

        :param inequalities: Optional parameter to list inequality constraints.
        :type inequalities: list of :class:`sympy.core.exp.Expr`.
        :param equalities: Optional parameter to list equality constraints.
        :type equalities: list of :class:`sympy.core.exp.Expr`.
        :param momentinequalities: Optional parameter of inequalities defined
                                   on moments.
        :type momentinequalities: list of :class:`sympy.core.exp.Expr`.
        :param momentequalities: Optional parameter of equalities defined
                                 on moments.
        :type momentequalities: list of :class:`sympy.core.exp.Expr`.
        :param removeequalities: Optional parameter to attempt removing the
                                 equalities by solving the linear equations.

        :param removeequalities: Optional parameter to attempt removing the
                                 equalities by solving the linear equations.
        :type removeequalities: bool.
        """
        self.status = "unsolved"
        if block_index == 0:
            if self._original_F is not None:
                self.F = self._original_F
                self.obj_facvar = self._original_obj_facvar
                self.constant_term = self._original_constant_term
                self.n_vars = len(self.obj_facvar)
                self._new_basis = None
            block_index = self.constraint_starting_block
            self.__wipe_F_from_constraints()
        self.constraints = flatten([inequalities])
        self._constraint_to_block_index = {}
        for constraint in self.constraints:
            self._constraint_to_block_index[constraint] = (block_index, )
            block_index += 1
        if momentinequalities is not None:
            for mineq in momentinequalities:
                self.constraints.append(mineq)
                self._constraint_to_block_index[mineq] = (block_index, )
                block_index += 1
        if not (removeequalities or equalities is None):
            # Equalities are converted to pairs of inequalities
            for k, equality in enumerate(equalities):
                if equality.is_Relational:
                    equality = convert_relational(equality)
                self.constraints.append(equality)
                self.constraints.append(-equality)
                ln = len(self.localizing_monomial_sets[block_index-
                                                       self.constraint_starting_block])
                self._constraint_to_block_index[equality] = (block_index,
                                                             block_index+ln*(ln+1)//2)
                block_index += ln*(ln+1)
        if momentequalities is not None and not removeequalities:
            for meq in momentequalities:
                self.constraints += [meq, flip_sign(meq)]
                self._constraint_to_block_index[meq] = (block_index,
                                                        block_index+1)
                block_index += 2
        block_index = self.constraint_starting_block
        self.__process_inequalities(block_index)
        if removeequalities:
            self.__remove_equalities(equalities, momentequalities)