def get_payload(self):
        """Return Payload."""
        payload = bytes([self.node_id])
        payload += bytes([self.state])
        payload += bytes(self.current_position.raw)
        payload += bytes(self.target.raw)
        payload += bytes(self.current_position_fp1.raw)
        payload += bytes(self.current_position_fp2.raw)
        payload += bytes(self.current_position_fp3.raw)
        payload += bytes(self.current_position_fp4.raw)
        payload += bytes([self.remaining_time >> 8 & 255, self.remaining_time & 255])
        payload += struct.pack(">I", self.timestamp)
        return payload