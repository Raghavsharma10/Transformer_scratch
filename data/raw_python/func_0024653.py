def get_payload(self):
        """Return Payload."""
        return bytes(
            [self.major_version >> 8 & 255, self.major_version & 255,
             self.minor_version >> 8 & 255, self.minor_version & 255])