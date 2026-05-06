def get_payload(self):
        """Return Payload."""
        payload = bytes([self.gateway_state.value, self.gateway_sub_state.value])
        payload += bytes(4)  # State date, reserved for future use
        return payload