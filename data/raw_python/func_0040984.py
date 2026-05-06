def verify(self, email):
        """
        Verify a single email address.
        :param str email: Email address to verify.
        :return: A VerifiedEmail object.
        """
        resp = self._call(endpoint='single', data={'email': email})
        return VerifiedEmail(email, resp['result'])