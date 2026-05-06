def from_payload(self, payload):
        """Init frame from binary data."""
        self.session_id = payload[0]*256 + payload[1]
        self.status = CommandSendConfirmationStatus(payload[2])