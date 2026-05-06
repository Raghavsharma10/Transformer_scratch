def decryptMsg(self, ciphertext, textMsg=True):
        """
        :type ciphertext: WhisperMessage
        :type textMsg: Bool set this to False if you are decrypting bytes
                       instead of string
        """

        if not self.sessionStore.containsSession(self.recipientId, self.deviceId):
            raise NoSessionException("No session for: %s, %s" % (self.recipientId, self.deviceId))

        sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
        plaintext = self.decryptWithSessionRecord(sessionRecord, ciphertext)

        self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)

        return plaintext