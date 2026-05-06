def initializeSessionAsBob(sessionState, sessionVersion, parameters):
        """
        :type sessionState: SessionState
        :type sessionVersion: int
        :type parameters: BobAxolotlParameters
        """
        sessionState.setSessionVersion(sessionVersion)
        sessionState.setRemoteIdentityKey(parameters.getTheirIdentityKey())
        sessionState.setLocalIdentityKey(parameters.getOurIdentityKey().getPublicKey())

        secrets = bytearray()

        if sessionVersion >= 3:
            secrets.extend(RatchetingSession.getDiscontinuityBytes())

        secrets.extend(Curve.calculateAgreement(parameters.getTheirIdentityKey().getPublicKey(),
                                                parameters.getOurSignedPreKey().getPrivateKey()))

        secrets.extend(Curve.calculateAgreement(parameters.getTheirBaseKey(),
                                                parameters.getOurIdentityKey().getPrivateKey()))
        secrets.extend(Curve.calculateAgreement(parameters.getTheirBaseKey(),
                                                parameters.getOurSignedPreKey().getPrivateKey()))

        if sessionVersion >= 3 and parameters.getOurOneTimePreKey() is not None:
            secrets.extend(Curve.calculateAgreement(parameters.getTheirBaseKey(),
                                                    parameters.getOurOneTimePreKey().getPrivateKey()))

        derivedKeys = RatchetingSession.calculateDerivedKeys(sessionVersion, secrets)
        sessionState.setSenderChain(parameters.getOurRatchetKey(), derivedKeys.getChainKey())
        sessionState.setRootKey(derivedKeys.getRootKey())