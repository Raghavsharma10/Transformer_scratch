def initializeSession(sessionState, sessionVersion, parameters):
        """
        :type sessionState: SessionState
        :type sessionVersion: int
        :type parameters: SymmetricAxolotlParameters
        """
        if RatchetingSession.isAlice(parameters.getOurBaseKey().getPublicKey(), parameters.getTheirBaseKey()):
            aliceParameters = AliceAxolotlParameters.newBuilder()
            aliceParameters.setOurBaseKey(parameters.getOurBaseKey()) \
                .setOurIdentityKey(parameters.getOurIdentityKey()) \
                .setTheirRatchetKey(parameters.getTheirRatchetKey()) \
                .setTheirIdentityKey(parameters.getTheirIdentityKey()) \
                .setTheirSignedPreKey(parameters.getTheirBaseKey()) \
                .setTheirOneTimePreKey(None)
            RatchetingSession.initializeSessionAsAlice(sessionState, sessionVersion, aliceParameters.create())
        else:
            bobParameters = BobAxolotlParameters.newBuilder()
            bobParameters.setOurIdentityKey(parameters.getOurIdentityKey()) \
                .setOurRatchetKey(parameters.getOurRatchetKey()) \
                .setOurSignedPreKey(parameters.getOurBaseKey()) \
                .setOurOneTimePreKey(None) \
                .setTheirBaseKey(parameters.getTheirBaseKey()) \
                .setTheirIdentityKey(parameters.getTheirIdentityKey())
            RatchetingSession.initializeSessionAsBob(sessionState, sessionVersion, bobParameters.create())