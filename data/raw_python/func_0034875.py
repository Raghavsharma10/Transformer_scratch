def unwrap_message(self, message, signature):
        """
        Verifies the supplied signature against the message and decrypts the message if 'confidentiality' was
        negotiated.
        A SignatureException is raised if the signature cannot be parsed or the version is unsupported
        A SequenceException is raised if the sequence number in the signature is incorrect
        A ChecksumException is raised if the in the signature checksum is invalid
        :return: The decrypted message
        """
        if not self.is_established:
            raise Exception("Context has not been established")
        if self._wrapper is None:
            raise Exception("Neither sealing or signing have been negotiated")
        else:
            return self._wrapper.unwrap(message, signature)