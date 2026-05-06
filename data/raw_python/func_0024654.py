def from_payload(self, payload):
        """Init frame from binary data."""
        self.major_version = payload[0] * 256 + payload[1]
        self.minor_version = payload[2] * 256 + payload[3]