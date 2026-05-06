def from_payload(self, payload):
        """Init frame from binary data."""
        self.session_id = payload[0]*256 + payload[1]
        self.originator = Originator(payload[2])
        self.priority = Priority(payload[3])
        self.scene_id = payload[4]
        self.velocity = Velocity(payload[5])