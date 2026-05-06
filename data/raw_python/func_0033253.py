def nonparabolicity(self, **kwargs):
        '''
        Returns the Kane band nonparabolicity parameter for the Gamma-valley.
        '''
        Eg = self.Eg_Gamma(**kwargs)
        meff = self.meff_e_Gamma(**kwargs)
        T = kwargs.get('T', 300.)
        return k*T/Eg * (1 - meff)**2