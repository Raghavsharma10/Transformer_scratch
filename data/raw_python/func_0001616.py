def sign(self, payload):
        """ Sign payload using the supplied authenticator """
        if self.authenticator:
            return self.authenticator.signed(payload)
        return payload