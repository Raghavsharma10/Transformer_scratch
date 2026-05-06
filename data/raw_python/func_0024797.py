def from_payload(self, payload):
        """Init frame from binary data."""
        self.status = ActivateSceneConfirmationStatus(payload[0])
        self.session_id = payload[1]*256 + payload[2]