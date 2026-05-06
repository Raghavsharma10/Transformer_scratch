def solve_gamlasso(self, lam):
        '''Solves the Graph-fused gamma lasso via POSE (Taddy, 2013)'''
        weights = lam / (1 + self.gamma * np.abs(self.beta[self.trails[::2]] - self.beta[self.trails[1::2]]))
        s = self.solve_gfl(u)
        self.steps.append(s)
        return self.beta