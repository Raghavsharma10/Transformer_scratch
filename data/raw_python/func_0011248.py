def get_confidence(self):
        """return confidence based on existing data"""
        # if we didn't receive any character in our consideration range,
        # return negative answer
        if self._mTotalChars <= 0 or self._mFreqChars <= MINIMUM_DATA_THRESHOLD:
            return SURE_NO

        if self._mTotalChars != self._mFreqChars:
            r = (self._mFreqChars / ((self._mTotalChars - self._mFreqChars)
                 * self._mTypicalDistributionRatio))
            if r < SURE_YES:
                return r

        # normalize confidence (we don't want to be 100% sure)
        return SURE_YES