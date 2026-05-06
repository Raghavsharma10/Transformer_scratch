def sign(self, encoded):
        """ Return authentication signature of encoded bytes """
        signature = self._hmac.copy()
        signature.update(encoded)
        return signature.hexdigest().encode('utf-8')