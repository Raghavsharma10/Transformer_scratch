def CBO_L(self, **kwargs):
        '''
        Returns the strain-shifted L-valley conduction band offset (CBO),
        assuming the strain affects all conduction band valleys equally.
        '''
        return (self.unstrained.CBO_L(**kwargs) +
                self.CBO_strain_shift(**kwargs))