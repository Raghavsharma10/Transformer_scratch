def getSignature(self, signatureKey, serialized):
        """
        :type signatureKey: ECPrivateKey
        :type serialized: bytearray
        """
        try:
            return Curve.calculateSignature(signatureKey, serialized)
        except InvalidKeyException as e:
            raise AssertionError(e)