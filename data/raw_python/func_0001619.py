def encode(self, payload):
        """ Encode payload """
        try:
            return self.encoder.encode(payload)
        except Exception as exception:
            raise EncodeError(str(exception))