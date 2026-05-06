def mask(self):
        '''
        The array of indices to be masked. This is the union of the sets of
        outliers, bad (flagged) cadences, transit cadences, and :py:obj:`NaN`
        cadences.

        '''

        return np.array(list(set(np.concatenate([self.outmask, self.badmask,
                        self.transitmask, self.nanmask]))), dtype=int)