def removeFixedEffect(self, index=None):
        """
        set sample and trait designs
        F:      NxK sample design
        A:      LxP sample design
        REML:   REML for this term?
        index:  index of which fixed effect to replace. If None, remove last term.
        """
        if self._n_terms==0:
            pass
        if index is None or index==(self._n_terms-1):

            self._n_terms-=1
            F = self._F.pop() #= self.F[:-1]
            A = self._A.pop() #= self.A[:-1]
            self._A_identity.pop() #= self.A_identity[:-1]
            REML_term = self._REML_term.pop()# = self.REML_term[:-1]
            self._B.pop()# = self.B[:-1]
            self._n_fixed_effs-=F.shape[1]*A.shape[0]
            if REML_term:
                self._n_fixed_effs_REML-=F.shape[1]*A.shape[0]

            pass
        elif index >= self.n_terms:
            raise Exception("index exceeds max index of terms")
        else:
            raise NotImplementedError("currently only last term can be removed")
            pass
        self._rebuild_indicator()
        self.clear_cache('Fstar','Astar','Xstar','Xhat',
                         'Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')