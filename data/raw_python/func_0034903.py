def Lc(self,value):
        """ set col rotation """
        assert value.shape[0]==self._P, 'Lc dimension mismatch'
        assert value.shape[1]==self._P, 'Lc dimension mismatch'
        self._Lc = value
        self.clear_cache('Astar','Ystar','Yhat','Xstar','Xhat',
                         'Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')