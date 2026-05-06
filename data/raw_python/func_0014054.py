def decryptWithSessionRecord(self, sessionRecord, cipherText):
        """
        :type sessionRecord: SessionRecord
        :type cipherText: WhisperMessage
        """

        previousStates = sessionRecord.getPreviousSessionStates()
        exceptions = []
        try:
            sessionState = SessionState(sessionRecord.getSessionState())
            plaintext = self.decryptWithSessionState(sessionState, cipherText)
            sessionRecord.setState(sessionState)
            return plaintext
        except InvalidMessageException as e:
            exceptions.append(e)

        for i in range(0, len(previousStates)):
            previousState = previousStates[i]
            try:
                promotedState = SessionState(previousState)
                plaintext = self.decryptWithSessionState(promotedState, cipherText)
                previousStates.pop(i)
                sessionRecord.promoteState(promotedState)
                return plaintext
            except InvalidMessageException as e:
                exceptions.append(e)

        raise InvalidMessageException("No valid sessions", exceptions)