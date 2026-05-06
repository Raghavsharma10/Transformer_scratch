def from_payload(self, payload):
        """Init frame from binary data."""
        self.session_id = payload[0]*256 + payload[1]
        self.index_id = payload[2]
        self.node_parameter = payload[3]
        self.seconds = payload[4]*256 + payload[5]