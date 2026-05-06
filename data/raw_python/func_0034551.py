def unwrap(self, message, signature):
        """
        NTLM GSSUnwrap()
        :param message: The message to be encrypted
        :return: The signed and encrypted message
        """
        plain_text = _Ntlm1Session.decrypt(self, message)
        _Ntlm1Session.verify(self, plain_text, signature)
        return plain_text