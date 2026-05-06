def wrap_message(self, message):
        """
        Cryptographically signs and optionally encrypts the supplied message. The message is only encrypted if
        'confidentiality' was negotiated, otherwise the message is left untouched.
        :return: A tuple containing the message signature and the optionally encrypted message
        """
        if not self.is_established:
            raise Exception("Context has not been established")
        if self._wrapper is None:
            raise Exception("Neither sealing or signing have been negotiated")
        else:
            return self._wrapper.wrap(message)