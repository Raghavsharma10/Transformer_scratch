def Lr(self,value):
        """ set row rotation """
        assert value.shape[0]==self._N, 'dimension mismatch'
        assert value.shape[1]==self._N, 'dimension mismatch'
        self._Lr = value
        self.clear_cache('Fstar','Ystar1','Ystar','Yhat','Xstar','Xhat',
                         'Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')