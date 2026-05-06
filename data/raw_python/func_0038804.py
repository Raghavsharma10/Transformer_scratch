def transform(self, X, y=None):
        '''
        :param X: list of dict which contains metabolic measurements.
        '''
        return Parallel(n_jobs=self.n_jobs)(delayed(self._transform)(x)
                                            for x in X)