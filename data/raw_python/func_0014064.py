def setUnacknowledgedPreKeyMessage(self, preKeyId, signedPreKeyId, baseKey):
        """
        :type preKeyId: int
        :type signedPreKeyId: int
        :type baseKey: ECPublicKey
        """
        self.sessionStructure.pendingPreKey.signedPreKeyId = signedPreKeyId
        self.sessionStructure.pendingPreKey.baseKey = baseKey.serialize()

        if preKeyId is not None:
            self.sessionStructure.pendingPreKey.preKeyId = preKeyId