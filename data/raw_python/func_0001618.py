def decode(self, payload):
        """ Decode payload """
        try:
            return self.encoder.decode(payload)
        except Exception as exception:
            raise DecodeError(str(exception))