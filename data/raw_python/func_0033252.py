def meff_lh_110(self, **kwargs):
        '''
        Returns the light-hole band effective mass in the [110] direction,
        meff_lh_110, in units of electron mass.
        '''
        return 2. / (2 * self.luttinger1(**kwargs) + self.luttinger2(**kwargs)
                + 3 * self.luttinger3(**kwargs))