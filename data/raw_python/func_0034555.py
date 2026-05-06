def wrap(self, message):
        """
        NTM GSSwrap()
        :param message: The message to be encrypted
        :return: A Tuple containing the signature and the encrypted messaging
        """
        cipher_text = _Ntlm2Session.encrypt(self, message)
        signature = _Ntlm2Session.sign(self, message)
        return cipher_text, signature