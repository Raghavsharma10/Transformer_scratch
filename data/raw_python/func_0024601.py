def from_payload(self, payload):
        """Init frame from binary data."""
        self.gateway_state = GatewayState(payload[0])
        self.gateway_sub_state = GatewaySubState(payload[1])