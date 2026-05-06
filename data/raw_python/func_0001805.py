def auth(self, encoded):
        """ Validate integrity of encoded bytes """
        message, signature = self.split(encoded)
        computed = self.sign(message)
        if not hmac.compare_digest(signature, computed):
            raise AuthenticatorInvalidSignature