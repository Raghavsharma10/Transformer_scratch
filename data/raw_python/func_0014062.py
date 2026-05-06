def getCipherText(self, iv, key, plaintext):
        """
        :type iv: bytearray
        :type key: bytearray
        :type plaintext: bytearray
        """
        cipher = AESCipher(key, iv)
        return cipher.encrypt(bytes(plaintext))