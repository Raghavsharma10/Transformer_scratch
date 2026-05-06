def solve(self, b_any, b, check_finite=True, p=None):
        """
        solve A \ b
        """
        #assert b.shape[:2]==(len(self.solver),self.dof_any)
        

        if self.schur_solver is None and self.A_any_solver is None:
            assert ( (b is None) or (b.shape[0]==0) ) and ( (b_any is None) or (b_any.shape[0]==0) ), "shape missmatch"
            return b, b_any
        elif self.schur_solver is None:
            assert (b is None) or (b.shape[0]==0), "shape missmatch"
            solution_any = self.A_any_solver.solve(b=b_any,p=p)
            return b,solution_any
        elif self.A_any_solver is None:
            assert (b_any is None) or (b_any.shape[0]==0), "shape missmatch"
            solution = self.schur_solver.solve(b=b, check_finite=check_finite)
            return solution, b_any
        else:
            assert p is None, "p is not None"
            cross_term = np.tensordot(self.DinvC,b_any,axes=([0,1],[0,1]))
            solution = self.schur_solver.solve(b=(b - cross_term), check_finite=check_finite)
            solution_any = self.A_any_solver.solve(b=b_any, check_finite=check_finite, p=p)
            solution_any -= self.DinvC.dot(solution)
            return solution, solution_any