def CBO(self, **kwargs):
        '''
        Returns the strain-shifted conduction band offset (CBO), assuming
        the strain affects all conduction band valleys equally.
        '''
        return self.unstrained.CBO(**kwargs) + self.CBO_strain_shift(**kwargs)