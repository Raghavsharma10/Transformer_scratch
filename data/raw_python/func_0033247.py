def F(self, **kwargs):
        '''
        Returns the Kane remote-band parameter, `F`, calculated from
        `Eg_Gamma_0`, `Delta_SO`, `Ep`, and `meff_e_Gamma_0`.
        '''
        Eg = self.Eg_Gamma_0(**kwargs)
        Delta_SO = self.Delta_SO(**kwargs)
        Ep = self.Ep(**kwargs)
        meff = self.meff_e_Gamma_0(**kwargs)
        return (1./meff-1-(Ep*(Eg+2.*Delta_SO/3.))/(Eg*(Eg+Delta_SO)))/2