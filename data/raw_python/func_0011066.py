def get_relaxation(self, level, objective=None, inequalities=None,
                       equalities=None, substitutions=None,
                       momentinequalities=None, momentequalities=None,
                       momentsubstitutions=None,
                       removeequalities=False, extramonomials=None,
                       extramomentmatrices=None, extraobjexpr=None,
                       localizing_monomials=None, chordal_extension=False):
        """Get the SDP relaxation of a noncommutative polynomial optimization
        problem.

        :param level: The level of the relaxation. The value -1 will skip
                      automatic monomial generation and use only the monomials
                      supplied by the option `extramonomials`.
        :type level: int.
        :param obj: Optional parameter to describe the objective function.
        :type obj: :class:`sympy.core.exp.Expr`.
        :param inequalities: Optional parameter to list inequality constraints.
        :type inequalities: list of :class:`sympy.core.exp.Expr`.
        :param equalities: Optional parameter to list equality constraints.
        :type equalities: list of :class:`sympy.core.exp.Expr`.
        :param substitutions: Optional parameter containing monomials that can
                              be replaced (e.g., idempotent variables).
        :type substitutions: dict of :class:`sympy.core.exp.Expr`.
        :param momentinequalities: Optional parameter of inequalities defined
                                   on moments.
        :type momentinequalities: list of :class:`sympy.core.exp.Expr`.
        :param momentequalities: Optional parameter of equalities defined
                                 on moments.
        :type momentequalities: list of :class:`sympy.core.exp.Expr`.
        :param momentsubstitutions: Optional parameter containing moments that
                                    can be replaced.
        :type momentsubstitutions: dict of :class:`sympy.core.exp.Expr`.
        :param removeequalities: Optional parameter to attempt removing the
                                 equalities by solving the linear equations.
        :type removeequalities: bool.
        :param extramonomials: Optional paramter of monomials to be included,
                               on top of the requested level of relaxation.
        :type extramonomials: list of :class:`sympy.core.exp.Expr`.
        :param extramomentmatrices: Optional paramter of duplicating or adding
                               moment matrices.  A new moment matrix can be
                               unconstrained (""), a copy  of the first one
                               ("copy"), and satisfying a partial positivity
                               constraint ("ppt"). Each new moment matrix is
                               requested as a list of string of these options.
                               For instance, adding a single new moment matrix
                               as a copy of the first would be
                               ``extramomentmatrices=[["copy"]]``.
        :type extramomentmatrices: list of list of str.
        :param extraobjexpr: Optional parameter of a string expression of a
                             linear combination of moment matrix elements to be
                             included in the objective function.
        :type extraobjexpr: str.
        :param localizing_monomials: Optional parameter to specify sets of
                                     localizing monomials for each constraint.
                                     The internal order of constraints is
                                     inequalities first, followed by the
                                     equalities. If the parameter is specified,
                                     but for a certain constraint the automatic
                                     localization is requested, leave None in
                                     its place in this parameter.
        :type localizing_monomials: list of list of `sympy.core.exp.Expr`.
        :param chordal_extension: Optional parameter to request a sparse
                                  chordal extension.
        :type chordal_extension: bool.

        """
        if self.level < -1:
            raise Exception("Invalid level of relaxation")
        self.level = level
        if substitutions is None:
            self.substitutions = {}
        else:
            self.substitutions = substitutions
            for lhs, rhs in substitutions.items():
                if not is_pure_substitution_rule(lhs, rhs):
                    self.pure_substitution_rules = False
                if iscomplex(lhs) or iscomplex(rhs):
                    self.complex_matrix = True
        if momentsubstitutions is not None:
            self.moment_substitutions = momentsubstitutions.copy()
            # If we have a real-valued problem, the moment matrix is symmetric
            # and moment substitutions also apply to the conjugate monomials
            if not self.complex_matrix:
                for key, val in self.moment_substitutions.copy().items():
                    adjoint_monomial = apply_substitutions(key.adjoint(),
                                                           self.substitutions)
                    self.moment_substitutions[adjoint_monomial] = val
        if chordal_extension:
            self.variables = find_variable_cliques(self.variables, objective,
                                                   inequalities, equalities,
                                                   momentinequalities,
                                                   momentequalities)
        self.__generate_monomial_sets(extramonomials)
        self.localizing_monomial_sets = localizing_monomials

        # Figure out basic structure of the SDP
        self._calculate_block_structure(inequalities, equalities,
                                        momentinequalities, momentequalities,
                                        extramomentmatrices,
                                        removeequalities)
        self._estimate_n_vars()
        if extramomentmatrices is not None:
            for parameters in extramomentmatrices:
                copy = False
                for parameter in parameters:
                    if parameter == "copy":
                        copy = True
                if copy:
                    self.n_vars += self.n_vars + 1
                else:
                    self.n_vars += (self.block_struct[0]**2)/2
        if self.complex_matrix:
            dtype = np.complex128
        else:
            dtype = np.float64
        self.F = lil_matrix((sum([bs**2 for bs in self.block_struct]),
                                    self.n_vars + 1), dtype=dtype)

        if self.verbose > 0:
            print(('Estimated number of SDP variables: %d' % self.n_vars))
            print('Generating moment matrix...')
        # Generate moment matrices
        new_n_vars, block_index = self.__add_parameters()
        self._time0 = time.time()
        new_n_vars, block_index = \
            self._generate_all_moment_matrix_blocks(new_n_vars, block_index)
        if extramomentmatrices is not None:
            new_n_vars, block_index = \
                self.__add_extra_momentmatrices(extramomentmatrices,
                                                new_n_vars, block_index)
        # The initial estimate for the size of F was overly generous.
        self.n_vars = new_n_vars
        # We don't correct the size of F, because that would trigger
        # memory copies, and extra columns in lil_matrix are free anyway.
        # self.F = self.F[:, 0:self.n_vars + 1]

        if self.verbose > 0:
            print(('Reduced number of SDP variables: %d' % self.n_vars))
        # Objective function
        self.set_objective(objective, extraobjexpr)
        # Process constraints
        self.constraint_starting_block = block_index
        self.process_constraints(inequalities, equalities, momentinequalities,
                                 momentequalities, block_index,
                                 removeequalities)