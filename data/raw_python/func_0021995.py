def _check_challenge(cls, challenge, challenge_body):
        """
        Check that the challenge resource we got is the one we expected.
        """
        if challenge.uri != challenge_body.uri:
            raise errors.UnexpectedUpdate(challenge.uri)
        return challenge