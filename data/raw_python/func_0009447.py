def relativeAreaSTE(self):
        '''
        return STE area - relative to image area
        '''
        s = self.noSTE.shape
        return np.sum(self.mask_STE) / (s[0] * s[1])