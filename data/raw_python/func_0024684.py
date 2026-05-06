def from_payload(self, payload):
        """Init frame from binary data."""
        self.status = NodeInformationStatus(payload[0])
        self.node_id = payload[1]