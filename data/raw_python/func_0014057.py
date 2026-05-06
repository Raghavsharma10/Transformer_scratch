def verifySignature(ecPublicSigningKey, message, signature):
        """
        :type ecPublicSigningKey: ECPublicKey
        :type message: bytearray
        :type signature: bytearray
        """

        if ecPublicSigningKey.getType() == Curve.DJB_TYPE:
            result = _curve.verifySignature(ecPublicSigningKey.getPublicKey(), message, signature)
            return result == 0
        else:
            raise InvalidKeyException("Unknown type: %s" % ecPublicSigningKey.getType())