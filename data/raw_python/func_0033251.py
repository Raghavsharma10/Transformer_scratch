def meff_hh_110(self, **kwargs):
        '''
        Returns the heavy-hole band effective mass in the [110] direction,
        meff_hh_110, in units of electron mass.
        '''
        return 2. / (2 * self.luttinger1(**kwargs) - self.luttinger2(**kwargs)
                - 3 * self.luttinger3(**kwargs))