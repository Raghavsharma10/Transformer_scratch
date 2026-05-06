def from_payload(self, payload):
        """Init frame from binary data."""
        self.status = AllNodesInformationStatus(payload[0])
        self.number_of_nodes = payload[1]