def _rebuild_indicator(self):
        """ update the indicator """
        indicator = {'term':np.array([]),
                     'row':np.array([]),
                     'col':np.array([])}

        for term in range(self.n_terms):
            L = self.A[term].shape[0]
            K = self.F[term].shape[1]
            _update = {'term': (term+1)*np.ones((K,L)).T.ravel(),
                    'row': np.kron(np.arange(K)[:,np.newaxis],np.ones((1,L))).T.ravel(),
                    'col': np.kron(np.ones((K,1)),np.arange(L)[np.newaxis,:]).T.ravel()}
            for key in list(_update.keys()):
                indicator[key] = np.concatenate([indicator[key],_update[key]])
        self.indicator = indicator