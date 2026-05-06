def setDesigns(self, F, A):
        """ set fixed effect designs """
        F = to_list(F)
        A = to_list(A)
        assert len(A) == len(F), 'MeanKronSum: A and F must have same length!'
        n_terms = len(F)
        n_covs = 0
        k = 0
        l = 0
        for ti in range(n_terms):
            assert F[ti].shape[0] == self._N, 'MeanKronSum: Dimension mismatch'
            assert A[ti].shape[1] == self._P, 'MeanKronSum: Dimension mismatch'
            n_covs += F[ti].shape[1] * A[ti].shape[0]
            k += F[ti].shape[1]
            l += A[ti].shape[0]
        self._n_terms = n_terms
        self._n_covs = n_covs
        self._k = k
        self._l = l
        self._F = F
        self._A = A
        self._b = sp.zeros((n_covs, 1))
        self.clear_cache('predict_in_sample', 'Yres', 'designs')
        self._notify('designs')
        self._notify()