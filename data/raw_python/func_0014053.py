def decryptPkmsg(self, ciphertext, textMsg=True):
        """
        :type ciphertext: PreKeyWhisperMessage
        """
        sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
        unsignedPreKeyId = self.sessionBuilder.process(sessionRecord, ciphertext)
        plaintext = self.decryptWithSessionRecord(sessionRecord, ciphertext.getWhisperMessage())

        # callback.handlePlaintext(plaintext)
        self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)

        if unsignedPreKeyId is not None:
            self.preKeyStore.removePreKey(unsignedPreKeyId)

        return plaintext