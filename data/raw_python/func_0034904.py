def d(self,value):
        """ set anisotropic scaling """
        assert value.shape[0]==self._P*self._N, 'd dimension mismatch'
        self._d = value
        self.clear_cache('Yhat','Xhat','Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')