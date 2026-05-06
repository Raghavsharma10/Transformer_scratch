def solve_t(self, Mt):
        """
        Mt is dim_r x dim_c x d tensor
        """
        if len(Mt.shape)==2:    _Mt = Mt[:, :, sp.newaxis]
        else:                   _Mt = Mt
        LMt = vei_CoR_veX(_Mt, R=self.Lr(), C=self.Lc())
        DLMt = self.D()[:, :, sp.newaxis] * LMt
        RV = vei_CoR_veX(DLMt, R=self.Lr().T, C=self.Lc().T)
        if len(Mt.shape)==2:    RV = RV[:, :, 0]
        return RV