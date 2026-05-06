def setSenderKeyState(self, id, iteration, chainKey, signatureKey):
        """
        :type id: int
        :type iteration: int
        :type chainKey: bytearray
        :type signatureKey: ECKeyPair
        """
        del self.senderKeyStates[:]
        self.senderKeyStates.append(SenderKeyState(id, iteration, chainKey, signatureKeyPair=signatureKey))