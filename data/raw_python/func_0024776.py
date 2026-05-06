def get_payload(self):
        """Return Payload."""
        ret = bytes([self.node_id])
        ret += string_to_bytes(self.name, 64)
        return ret