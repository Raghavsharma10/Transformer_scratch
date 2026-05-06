def solve_dp(self, lam):
        '''Solves the Graph-fused double Pareto (non-convex, local optima only)'''
        cur_converge = self.converge+1
        step = 0
        # Get an initial estimate using the GFL
        self.solve_gfl(lam)
        beta2 = np.copy(self.beta)
        while cur_converge > self.converge and step < self.max_dp_steps:
            # Weight each edge differently
            u = lam / (1 + np.abs(self.beta[self.trails[::2]] - self.beta[self.trails[1::2]]))
            # Swap the beta buffers
            temp = self.beta
            self.beta = beta2
            beta2 = temp
            # Solve the edge-weighted GFL problem, which updates beta
            self.solve_gfl(u)
            # Check for convergence
            cur_converge = np.sqrt(((self.beta - beta2)**2).sum())
            step += 1
        self.steps.append(step)
        return self.beta