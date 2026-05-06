def verify(self, payload):
        """ Verify payload authenticity via the supplied authenticator """
        if not self.authenticator:
            return payload
        try:
            self.authenticator.auth(payload)
            return self.authenticator.unsigned(payload)
        except AuthenticatorInvalidSignature:
            raise
        except Exception as exception:
            raise AuthenticateError(str(exception))