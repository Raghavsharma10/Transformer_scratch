def addFixedEffect(self,F=None,A=None, REML=True, index=None):
        """
        set sample and trait designs
        F:      NxK sample design
        A:      LxP sample design
        REML:   REML for this term?
        index:  index of which fixed effect to replace. If None, just append.
        """
        if F is None:   F = np.ones((self.N,1))
        if A is None:
            A = np.eye(self.P)
            A_identity = True
        elif (A.shape == (self.P,self.P)) & (A==np.eye(self.P)).all():
            A_identity = True
        else:
            A_identity = False

        assert F.shape[0]==self.N, "F dimension mismatch"
        assert A.shape[1]==self.P, "A dimension mismatch"
        if index is None or index==self.n_terms:
            self.F.append(F)
            self.A.append(A)
            self.A_identity.append(A_identity)
            self.REML_term.append(REML)
            # build B matrix and indicator
            self.B.append(np.zeros((F.shape[1],A.shape[0])))
            self._n_terms+=1
            self._update_indicator(F.shape[1],A.shape[0])
        elif index >self.n_terms:
            raise Exception("index exceeds max index of terms")
        else:
            self._n_fixed_effs-=self.F[index].shape[1]*self.A[index].shape[0]
            if self.REML_term[index]:
                self._n_fixed_effs_REML-=self.F[index].shape[1]*self.A[index].shape[0]
            self.F[index] = F
            self.A[index] = A
            self.A_identity[index] = A_identity
            self.REML_term[index]=REML
            self.B[index] = np.zeros((F.shape[1],A.shape[0]))
            self._rebuild_indicator()

        self._n_fixed_effs+=F.shape[1]*A.shape[0]
        if REML:
            self._n_fixed_effs_REML+=F.shape[1]*A.shape[0]
        self.clear_cache('Fstar','Astar','Xstar','Xhat',
                         'Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')