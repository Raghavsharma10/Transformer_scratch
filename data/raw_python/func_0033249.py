def meff_SO(self, **kwargs):
        '''
        Returns the split-off hole effective mass
        calculated from Eg_Gamma(T), Delta_SO, Ep and F.
        
        Interpolation of Eg_Gamma(T), Delta_SO, Ep and luttinger1, and
        then calculation of meff_SO is recommended for alloys.
        '''
        Eg = self.Eg_Gamma(**kwargs)
        Delta_SO = self.Delta_SO(**kwargs)
        Ep = self.Ep(**kwargs)
        luttinger1 = self.luttinger1(**kwargs)
        return 1./(luttinger1 - (Ep*Delta_SO)/(3*Eg*(Eg+Delta_SO)))