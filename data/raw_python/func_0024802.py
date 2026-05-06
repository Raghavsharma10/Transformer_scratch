def from_payload(self, payload):
        """Init frame from binary data."""
        self.session_id = payload[0]*256 + payload[1]
        self.status_id = payload[2]
        self.index_id = payload[3]
        self.node_parameter = payload[4]
        self.parameter_value = payload[5]*256 + payload[6]