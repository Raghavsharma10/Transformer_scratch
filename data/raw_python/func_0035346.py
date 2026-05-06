def Kgrad_param_num(self,i,h=1e-4):
        """
        check discrepancies between numerical and analytical gradients
        """
        params = self.getParams()
        e = sp.zeros_like(params); e[i] = 1
        self.setParams(params-h*e)
        C_L = self.K()
        self.setParams(params+h*e)
        C_R = self.K()
        self.setParams(params)
        RV = (C_R-C_L)/(2*h)
        return RV