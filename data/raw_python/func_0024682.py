def get_payload(self):
        """Return Payload."""
        payload = bytes([self.node_id])
        payload += string_to_bytes(self.name, 64)
        payload += bytes([self.order >> 8 & 255, self.order & 255])
        payload += bytes([self.placement])
        payload += bytes([self.node_variation.value])
        return payload