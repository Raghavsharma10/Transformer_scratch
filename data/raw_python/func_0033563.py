def Eg(self, **kwargs):
        '''
        Returns the strain-shifted bandgap, ``Eg``.
        '''
        return self.unstrained.Eg(**kwargs) + self.Eg_strain_shift(**kwargs)