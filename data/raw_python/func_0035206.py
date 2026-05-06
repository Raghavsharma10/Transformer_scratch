def addFixedEffect(self,F=None,A=None,index=None):
        """
        set sample and trait designs
        F:      NxK sample design
        A:      LxP sample design
        fast_computations:   False deactivates the fast computations for any and common effects (for debugging)
        """
        if F is None:
            F = np.ones((self.N,1))
        else:
            assert F.shape[0]==self.N, "F dimension mismatch"

        if ((A is None) or ( (A.shape == (self.P,self.P)) and (A==np.eye(self.P)).all() )):
            #case any effect
            self.F_any = np.hstack((self.F_any,F))
        elif (index is not None) and  ((A==self.A[index]).all()):
            #case common effect
            self.F[index] = np.hstack((self.F_index,F))
        else:
            #case general A
            assert A.shape[1]==self.P, "A dimension mismatch"
            self.F.append(F)
            self.A.append(A)

        self.clear_cache()