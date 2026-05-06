def CBO_Gamma(self, **kwargs):
        '''
        Returns the strain-shifted Gamma-valley conduction band offset (CBO),
        assuming the strain affects all conduction band valleys equally.
        '''
        return (self.unstrained.CBO_Gamma(**kwargs) +
                self.CBO_strain_shift(**kwargs))