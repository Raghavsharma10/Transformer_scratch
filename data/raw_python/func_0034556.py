def unwrap(self, message, signature):
        """
        NTLM GSSUnwrap()
        :param message: The message to be decrypted
        :return: The decrypted message
        """
        plain_text = _Ntlm2Session.decrypt(self, message)
        _Ntlm2Session.verify(self, plain_text, signature)
        return plain_text