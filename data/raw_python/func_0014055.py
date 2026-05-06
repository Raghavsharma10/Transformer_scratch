def getCiphertext(self, version, messageKeys, plainText):
        """
        :type version: int
        :type messageKeys: MessageKeys
        :type  plainText: bytearray
        """
        cipher = None
        if version >= 3:
            cipher = self.getCipher(messageKeys.getCipherKey(), messageKeys.getIv())
        else:
            cipher = self.getCipher_v2(messageKeys.getCipherKey(), messageKeys.getCounter())

        return cipher.encrypt(bytes(plainText))