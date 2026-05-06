def feed(self, aBuf, aCharLen):
        """feed a character with known length"""
        if aCharLen == 2:
            # we only care about 2-bytes character in our distribution analysis
            order = self.get_order(aBuf)
        else:
            order = -1
        if order >= 0:
            self._mTotalChars += 1
            # order is valid
            if order < self._mTableSize:
                if 512 > self._mCharToFreqOrder[order]:
                    self._mFreqChars += 1