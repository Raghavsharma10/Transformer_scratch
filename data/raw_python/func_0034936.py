def solve(self,b,overwrite_b=False,check_finite=True, p=None):
        """
        solve A \ b
        """
        if p is None:
            assert b.shape[:2]==(len(self.solver),self.dof_any)
            solution = np.empty(b.shape)
            #This is trivially parallelizable:
            for p in range(self.P):
                solution[p] = self.solver[p].solve(b=b[p])
            return solution
        else:
            return self.solver[p].solve(b=b)