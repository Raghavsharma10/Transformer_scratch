def get_payload(self):
        """Return Payload."""
        ret = bytes([self.session_id >> 8 & 255, self.session_id & 255])
        ret += bytes([self.status_id])
        ret += bytes([self.index_id])
        ret += bytes([self.node_parameter])
        ret += bytes([self.parameter_value >> 8 & 255, self.parameter_value & 255])

        # XXX: Missing implementation of run_status, status_reply and information_code
        ret += bytes(6)
        return ret