def get_payload(self):
        """Return Payload."""
        ret = bytes([self.session_id >> 8 & 255, self.session_id & 255])
        ret += bytes([self.index_id])
        ret += bytes([self.node_parameter])
        ret += bytes([self.seconds >> 8 & 255, self.seconds & 255])
        return ret