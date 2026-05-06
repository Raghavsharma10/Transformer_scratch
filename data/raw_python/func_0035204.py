def dof(self, index=None):
        """The number of degrees of freedom"""
        if index is None:
            dof = 0
            for i in range(self.len):
                dof += self.A[i].shape[0] * self.F[i].shape[1]
            return dof
        else:
            return self.A[index].shape[0] * self.F[index].shape[1]