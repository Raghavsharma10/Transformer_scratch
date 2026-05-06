def solve_t(self, Mt):
        """
        Mt is dim_r x dim_c x d tensor
        """
        if len(Mt.shape)==2:    _Mt = Mt[:, :, sp.newaxis]
        else:                   _Mt = Mt
        M = _Mt.transpose([0,2,1])
        MLc = sp.tensordot(M, self.Lc().T, (2,0)) 
        MLcLc = sp.tensordot(MLc, self.Lc(), (2,0)) 
        WrMLcWc = sp.tensordot(sp.tensordot(self.Wr(), MLc, (1,0)), self.Wc().T, (2,0))
        DWrMLcWc = sp.tensordot(self.D()[:,sp.newaxis,:]*WrMLcWc, self.Wc(), (2,0))
        WrDWrMLcWcLc = sp.tensordot(self.Wr().T, sp.tensordot(DWrMLcWc, self.Lc(), (2,0)), (1,0))
        RV = (MLcLc - WrDWrMLcWcLc).transpose([0,2,1])
        if len(Mt.shape)==2:    RV = RV[:, :, 0]
        return RV