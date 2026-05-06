def set_objective(self, objective, extraobjexpr=None):
        """Set or change the objective function of the polynomial optimization
        problem.

        :param objective: Describes the objective function.
        :type objective: :class:`sympy.core.expr.Expr`
        :param extraobjexpr: Optional parameter of a string expression of a
                             linear combination of moment matrix elements to be
                             included in the objective function
        :type extraobjexpr: str.
        """
        if objective is not None:
            facvar = \
                self._get_facvar(simplify_polynomial(objective,
                                                     self.substitutions))
            self.obj_facvar = facvar[1:]
            self.constant_term = facvar[0]
            if self.verbose > 0 and facvar[0] != 0:
                print("Warning: The objective function has a non-zero %s "
                      "constant term. It is not included in the SDP objective."
                      % facvar[0], file=sys.stderr)
        else:
            self.obj_facvar = self._get_facvar(0)[1:]
        if extraobjexpr is not None:
            for sub_expr in extraobjexpr.split(']'):
                startindex = 0
                if sub_expr.startswith('-') or sub_expr.startswith('+'):
                    startindex = 1
                ind = sub_expr.find('[')
                if ind > -1:
                    idx = sub_expr[ind+1:].split(",")
                    i, j = int(idx[0]), int(idx[1])
                    mm_ind = int(sub_expr[startindex:ind])
                    if sub_expr.find('*') > -1:
                        value = float(sub_expr[:sub_expr.find('*')])
                    elif sub_expr.startswith('-'):
                        value = -1.0
                    else:
                        value = 1.0
                    base_row_offset = sum([bs**2 for bs in
                                           self.block_struct[:mm_ind]])
                    width = self.block_struct[mm_ind]
                    for column in self.F[base_row_offset + i*width + j].rows[0]:
                        self.obj_facvar[column-1] = \
                            value*self.F[base_row_offset + i*width + j, column]