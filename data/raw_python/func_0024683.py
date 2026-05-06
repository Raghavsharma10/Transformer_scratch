def from_payload(self, payload):
        """Init frame from binary data."""
        self.node_id = payload[0]
        self.name = bytes_to_string(payload[1:65])
        self.order = payload[65] * 256 + payload[66]
        self.placement = payload[67]
        self.node_variation = NodeVariation(payload[68])