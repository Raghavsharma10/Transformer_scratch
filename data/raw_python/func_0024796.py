def get_payload(self):
        """Return Payload."""
        ret = bytes([self.status.value])
        ret += bytes([self.session_id >> 8 & 255, self.session_id & 255])
        return ret