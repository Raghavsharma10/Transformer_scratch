def b64encoded(self):
        """Return a base64 encoding of the key.

        returns:
            str: base64 encoding of the public key
        """
        if self._b64encoded:
            return text_type(self._b64encoded).strip("\r\n")
        else:
            return base64encode(self.raw)