def from_payload(self, payload):
        """Init frame from binary data."""
        self.node_id = payload[0]
        self.name = bytes_to_string(payload[1:65])