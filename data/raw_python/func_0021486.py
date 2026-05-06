def solve(self, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        s = weighted_graphtf_poisson(self.nnodes, self.obs, lam,
                                 self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                                 self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                                 self.maxsteps, self.converge,
                                 self.beta, self.u)
        self.steps.append(s)
        return self.beta