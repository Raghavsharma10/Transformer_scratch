def generateOneTimePad(self, userStore):
        """
        Generate a pad which can be used to authenticate via AMP.  This pad
        will expire in L{ONE_TIME_PAD_DURATION} seconds.
        """
        pad = secureRandom(16).encode('hex')
        self._oneTimePads[pad] = userStore.idInParent
        def expirePad():
            self._oneTimePads.pop(pad, None)
        self.callLater(self.ONE_TIME_PAD_DURATION, expirePad)
        return pad