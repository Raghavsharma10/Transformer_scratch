def get_payload(self):
        """Return Payload."""
        if self.password is None:
            raise PyVLXException("password is none")
        if len(self.password) > self.MAX_SIZE:
            raise PyVLXException("password is too long")
        return string_to_bytes(self.password, self.MAX_SIZE)