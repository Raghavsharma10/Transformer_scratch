def _update_indicator(self,K,L):
        """ update the indicator """
        _update = {'term': self.n_terms*np.ones((K,L)).T.ravel(),
                    'row': np.kron(np.arange(K)[:,np.newaxis],np.ones((1,L))).T.ravel(),
                    'col': np.kron(np.ones((K,1)),np.arange(L)[np.newaxis,:]).T.ravel()}
        for key in list(_update.keys()):
            self.indicator[key] = np.concatenate([self.indicator[key],_update[key]])