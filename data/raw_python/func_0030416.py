def determineFrom(cls, challenge, password):
        """
        Create a nonce and use it, along with the given challenge and password,
        to generate the parameters for a response.

        @return: A C{dict} suitable to be used as the keyword arguments when
            calling this command.
        """
        nonce = secureRandom(16)
        response = _calcResponse(challenge, nonce, password)
        return dict(cnonce=nonce, response=response)