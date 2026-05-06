def _scaleTo8bit(self, img):
        '''
        The pattern comparator need images to be 8 bit
        -> find the range of the signal and scale the image
        '''
        r = scaleSignalCutParams(img, 0.02)  # , nSigma=3)
        self.signal_ranges.append(r)
        return toUIntArray(img, dtype=np.uint8, range=r)