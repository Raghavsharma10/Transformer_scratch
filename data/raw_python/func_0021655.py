def solve(self, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        if self.penalty == 'dp':
            return self.solve_dp(lam)
        if self.penalty == 'gfl':
            return self.solve_gfl(lam)
        if self.penalty == 'gamlasso':
            return self.solve_gfl(lam)
        raise Exception('Unknown penalty type: {0}'.format(self.penalty))