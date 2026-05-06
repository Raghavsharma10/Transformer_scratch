def get_dual(self, constraint, ymat=None):
        """Given a solution of the dual problem and a constraint of any type,
        it returns the corresponding block in the dual solution. If it is an
        equality constraint that was converted to a pair of inequalities, it
        returns a two-tuple of the matching dual blocks.

        :param constraint: The constraint.
        :type index: `sympy.core.exp.Expr`
        :param y_mat: Optional parameter providing the dual solution of the
                      SDP. If not provided, the solution is extracted
                      from the sdpRelaxation object.
        :type y_mat: :class:`numpy.array`.
        :returns: The corresponding block in the dual solution.
        :rtype: :class:`numpy.array` or a tuple thereof.
        """
        if not isinstance(constraint, Expr):
            raise Exception("Not a monomial or polynomial!")
        elif self.status == "unsolved" and ymat is None:
            raise Exception("SDP relaxation is not solved yet!")
        elif ymat is None:
            ymat = self.y_mat
        index = self._constraint_to_block_index.get(constraint)
        if index is None:
            raise Exception("Constraint is not in the dual!")
        if len(index) == 2:
            return ymat[index[0]], self.y_mat[index[1]]
        else:
            return ymat[index[0]]