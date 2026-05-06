def clearFixedEffect(self):
        """ erase all fixed effects """
        self._A = []
        self._F = []
        self._B = []
        self._A_identity = []
        self._REML_term = []
        self._n_terms = 0
        self._n_fixed_effs = 0
        self._n_fixed_effs_REML = 0
        self.indicator = {'term':np.array([]),
                            'row':np.array([]),
                            'col':np.array([])}
        self.clear_cache('Fstar','Astar','Xstar','Xhat',
                         'Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')