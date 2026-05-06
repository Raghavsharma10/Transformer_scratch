def addSenderKeyState(self, id, iteration, chainKey, signatureKey):
        """
        :type id: int
        :type iteration: int
        :type chainKey: bytearray
        :type signatureKey: ECPublicKey
        """
        self.senderKeyStates.append(SenderKeyState(id, iteration, chainKey, signatureKey))