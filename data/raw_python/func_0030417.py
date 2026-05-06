def checkPassword(self, password):
        """
        Check the given plaintext password against the response in this
        credentials object.

        @type password: C{str}
        @param password: The known correct password associated with
            C{self.username}.

        @return: A C{bool}, C{True} if this credentials object agrees with the
            given password, C{False} otherwise.
        """
        if isinstance(password, unicode):
            password = password.encode('utf-8')
        correctResponse = _calcResponse(self.challenge, self.nonce, password)
        return correctResponse == self.response