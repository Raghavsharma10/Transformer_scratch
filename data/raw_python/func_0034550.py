def wrap(self, message):
        """
        NTM GSSwrap()
        :param message: The message to be encrypted
        :return: The signed and encrypted message
        """
        cipher_text = _Ntlm1Session.encrypt(self, message)
        signature = _Ntlm1Session.sign(self, message)
        return cipher_text, signature