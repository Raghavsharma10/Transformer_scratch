def getPlainText(self, iv, key, ciphertext):
        """
        :type iv: bytearray
        :type key: bytearray
        :type ciphertext: bytearray
        """
        try:
            cipher = AESCipher(key, iv)
            plaintext = cipher.decrypt(ciphertext)
            if sys.version_info >= (3, 0):
                return plaintext.decode()
            return plaintext
        except Exception as e:
            raise InvalidMessageException(e)