def processPreKeyBundle(self, preKey):
        """
        :type preKey: PreKeyBundle
        """
        if not self.identityKeyStore.isTrustedIdentity(self.recipientId, preKey.getIdentityKey()):
            raise UntrustedIdentityException(self.recipientId, preKey.getIdentityKey())

        if preKey.getSignedPreKey() is not None and\
            not Curve.verifySignature(preKey.getIdentityKey().getPublicKey(),
                                      preKey.getSignedPreKey().serialize(),
                                      preKey.getSignedPreKeySignature()):
            raise InvalidKeyException("Invalid signature on device key!")

        if preKey.getSignedPreKey() is None and preKey.getPreKey() is None:
            raise InvalidKeyException("Both signed and unsigned prekeys are absent!")

        supportsV3 = preKey.getSignedPreKey() is not None
        sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
        ourBaseKey = Curve.generateKeyPair()
        theirSignedPreKey = preKey.getSignedPreKey() if supportsV3 else preKey.getPreKey()
        theirOneTimePreKey = preKey.getPreKey()
        theirOneTimePreKeyId = preKey.getPreKeyId() if theirOneTimePreKey is not None else None

        parameters = AliceAxolotlParameters.newBuilder()

        parameters.setOurBaseKey(ourBaseKey)\
            .setOurIdentityKey(self.identityKeyStore.getIdentityKeyPair())\
            .setTheirIdentityKey(preKey.getIdentityKey())\
            .setTheirSignedPreKey(theirSignedPreKey)\
            .setTheirRatchetKey(theirSignedPreKey)\
            .setTheirOneTimePreKey(theirOneTimePreKey if supportsV3 else None)

        if not sessionRecord.isFresh():
            sessionRecord.archiveCurrentState()

        RatchetingSession.initializeSessionAsAlice(sessionRecord.getSessionState(),
                                                   3 if supportsV3 else 2,
                                                   parameters.create())

        sessionRecord.getSessionState().setUnacknowledgedPreKeyMessage(theirOneTimePreKeyId,
                                                                       preKey.getSignedPreKeyId(),
                                                                       ourBaseKey.getPublicKey())
        sessionRecord.getSessionState().setLocalRegistrationId(self.identityKeyStore.getLocalRegistrationId())
        sessionRecord.getSessionState().setRemoteRegistrationId(preKey.getRegistrationId())
        sessionRecord.getSessionState().setAliceBaseKey(ourBaseKey.getPublicKey().serialize())
        self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)
        self.identityKeyStore.saveIdentity(self.recipientId, preKey.getIdentityKey())