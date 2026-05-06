def Eg(self, **kwargs):
        '''
        Returns the bandgap, Eg, in eV at a given
        temperature, T, in K (default=300.).
        '''
        return min(self.Eg_Gamma(**kwargs),
                   self.Eg_L(**kwargs),
                   self.Eg_X(**kwargs))