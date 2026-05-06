def solve_t(self, Mt):
        """
        Mt is dim_r x dim_c x d tensor
        """
        if len(Mt.shape)==2:    _Mt = Mt[:, :, sp.newaxis]
        else:                   _Mt = Mt
        LMt = vei_CoR_veX(_Mt, R=self.Lr(), C=self.Lc())
        DMt = self.D()[:, :, sp.newaxis] * LMt
        WrDMtWc = vei_CoR_veX(DMt, R=self.Wr().T, C=self.Wc().T)
        ve_WrDMtWc = sp.reshape(WrDMtWc, (WrDMtWc.shape[0] * WrDMtWc.shape[1], _Mt.shape[2]), order='F')
        Hi_ve_WrDMtWc = la.cho_solve((self.H_chol(), True), ve_WrDMtWc)
        vei_HiveWrDMtWc = Hi_ve_WrDMtWc.reshape(WrDMtWc.shape, order = 'F')
        Wr_HiveWrDMtWc_Wc = vei_CoR_veX(vei_HiveWrDMtWc, R=self.Wr(), C=self.Wc())
        DWrHiveWrDMtWcWc = self.D()[:,:,sp.newaxis] * Wr_HiveWrDMtWc_Wc
        RV = DMt - DWrHiveWrDMtWcWc
        RV = vei_CoR_veX(RV, R=self.Lr().T, C=self.Lc().T)
        if len(Mt.shape)==2:    RV = RV[:, :, 0]
        return RV