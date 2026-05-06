def _O_dot(self, Mt):
        """
        Mt is dim_r x dim_c x d tensor
        """
        DMt = self.D()[:, :, sp.newaxis] * Mt
        WrDMtWc = vei_CoR_veX(DMt, R=self.Wr().T, C=self.Wc().T)
        ve_WrDMtWc = sp.reshape(WrDMtWc, (WrDMtWc.shape[0] * WrDMtWc.shape[1], Mt.shape[2]), order='F')
        Hi_ve_WrDMtWc = la.cho_solve((self.H_chol(), True), ve_WrDMtWc)
        vei_HiveWrDMtWc = Hi_ve_WrDMtWc.reshape(WrDMtWc.shape, order = 'F')
        Wr_HiveWrDMtWc_Wc = vei_CoR_veX(vei_HiveWrDMtWc, R=self.Wr(), C=self.Wc())
        DWrHiveWrDMtWcWc = self.D()[:,:,sp.newaxis] * Wr_HiveWrDMtWc_Wc
        RV = DMt - DWrHiveWrDMtWcWc
        return RV