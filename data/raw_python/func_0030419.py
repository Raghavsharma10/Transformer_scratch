def passwordChallengeResponse(self, cnonce, response):
        """
        Verify the response to a challenge.
        """
        return self._login(_AMPUsernamePassword(
            self.username, self.challenge, cnonce, response))