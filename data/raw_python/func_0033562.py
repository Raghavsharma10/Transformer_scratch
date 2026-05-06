def CBO_X(self, **kwargs):
        '''
        Returns the strain-shifted X-valley conduction band offset (CBO),
        assuming the strain affects all conduction band valleys equally.
        '''
        return (self.unstrained.CBO_X(**kwargs) +
                self.CBO_strain_shift(**kwargs))