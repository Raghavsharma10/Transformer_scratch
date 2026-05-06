def initialize_security_context(self):
        """
        Idiomatic Python implementation of initialize_security_context, implemented as a generator function using
        yield to both accept incoming and return outgoing authentication tokens
        :return: The response to be returned to the server
        """
        # Generate the NTLM Negotiate Request
        negotiate_token = self._negotiate(self.flags)
        challenge_token = yield negotiate_token

        # Generate the Authenticate Response
        authenticate_token = self._challenge_response(negotiate_token, challenge_token)
        yield authenticate_token