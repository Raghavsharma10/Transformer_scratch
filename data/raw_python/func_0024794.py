def get_payload(self):
        """Return Payload."""
        ret = bytes([self.session_id >> 8 & 255, self.session_id & 255])
        ret += bytes([self.originator.value])
        ret += bytes([self.priority.value])
        ret += bytes([self.scene_id])
        ret += bytes([self.velocity.value])
        return ret