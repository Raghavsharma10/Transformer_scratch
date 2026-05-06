def meff_e_Gamma(self, **kwargs):
        '''
        Returns the electron effective mass in the Gamma-valley
        calculated from Eg_Gamma(T), Delta_SO, Ep and F.
        
        Interpolation of Eg_Gamma(T), Delta_SO, Ep and F, and
        then calculation of meff_e_Gamma is recommended for alloys.
        '''
        Eg = self.Eg_Gamma(**kwargs)
        Delta_SO = self.Delta_SO(**kwargs)
        Ep = self.Ep(**kwargs)
        F = self.F(**kwargs)
        return 1./((1.+2.*F)+(Ep*(Eg+2.*Delta_SO/3.))/(Eg*(Eg+Delta_SO)))