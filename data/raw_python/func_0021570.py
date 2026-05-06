def data_log_likelihood(self, successes, trials, beta):
        '''Calculates the log-likelihood of a Polya tree bin given the beta values.'''
        return binom.logpmf(successes, trials, 1.0 / (1 + np.exp(-beta))).sum()