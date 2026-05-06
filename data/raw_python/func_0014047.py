def initializeSessionAsAlice(sessionState, sessionVersion, parameters):
        """
        :type sessionState: SessionState
        :type sessionVersion: int
        :type parameters: AliceAxolotlParameters
        """
        sessionState.setSessionVersion(sessionVersion)
        sessionState.setRemoteIdentityKey(parameters.getTheirIdentityKey())
        sessionState.setLocalIdentityKey(parameters.getOurIdentityKey().getPublicKey())

        sendingRatchetKey = Curve.generateKeyPair()
        secrets = bytearray()

        if sessionVersion >= 3:
            secrets.extend(RatchetingSession.getDiscontinuityBytes())

        secrets.extend(Curve.calculateAgreement(parameters.getTheirSignedPreKey(),
                                                parameters.getOurIdentityKey().getPrivateKey()))
        secrets.extend(Curve.calculateAgreement(parameters.getTheirIdentityKey().getPublicKey(),
                                                parameters.getOurBaseKey().getPrivateKey()))
        secrets.extend(Curve.calculateAgreement(parameters.getTheirSignedPreKey(),
                                                parameters.getOurBaseKey().getPrivateKey()))

        if sessionVersion >= 3 and parameters.getTheirOneTimePreKey() is not None:
            secrets.extend(Curve.calculateAgreement(parameters.getTheirOneTimePreKey(),
                                                    parameters.getOurBaseKey().getPrivateKey()))

        derivedKeys = RatchetingSession.calculateDerivedKeys(sessionVersion, secrets)
        sendingChain = derivedKeys.getRootKey().createChain(parameters.getTheirRatchetKey(), sendingRatchetKey)

        sessionState.addReceiverChain(parameters.getTheirRatchetKey(), derivedKeys.getChainKey())
        sessionState.setSenderChain(sendingRatchetKey, sendingChain[1])
        sessionState.setRootKey(sendingChain[0])