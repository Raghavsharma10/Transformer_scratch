def Y(self,value):
        """ set phenotype """
        self._N,self._P = value.shape
        self._Y = value
        self.clear_cache('Ystar1','Ystar','Yhat','LRLdiag_Yhat',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')