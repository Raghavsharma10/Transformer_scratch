def intensityDistributionSTE(self, bins=10, range=None):
        '''
        return distribution of STE intensity
        '''
        v = np.abs(self._last_diff[self.mask_STE])
        return np.histogram(v, bins, range)