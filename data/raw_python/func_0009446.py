def addImage(self, image, mask=None):
        '''
        #########
        mask -- optional
        '''
        self._last_diff = diff = image - self.noSTE

        ste = diff > self.threshold
        removeSinglePixels(ste)

        self.mask_clean = clean = ~ste

        if mask is not None:
            clean = np.logical_and(mask, clean)

        self.mma.update(image, clean)

        if self.save_ste_indices:
            self.mask_STE += ste

        return self