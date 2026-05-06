def process(self, sessionRecord, message):
        """
        :param sessionRecord:
        :param message:
        :type message: PreKeyWhisperMessage
        """

        messageVersion = message.getMessageVersion()
        theirIdentityKey = message.getIdentityKey()

        unsignedPreKeyId = None

        if not self.identityKeyStore.isTrustedIdentity(self.recipientId, theirIdentityKey):
            raise UntrustedIdentityException(self.recipientId, theirIdentityKey)

        if messageVersion == 2:
            unsignedPreKeyId = self.processV2(sessionRecord, message)
        elif messageVersion == 3:
            unsignedPreKeyId = self.processV3(sessionRecord, message)
        else:
            raise AssertionError("Unkown version %s" % messageVersion)

        self.identityKeyStore.saveIdentity(self.recipientId, theirIdentityKey)

        return unsignedPreKeyId