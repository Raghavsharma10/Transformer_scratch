def get_payload(self):
        """Return Payload."""
        ret = bytes([len(self.scenes)])
        for number, name in self.scenes:
            ret += bytes([number])
            ret += string_to_bytes(name, 64)
        ret += bytes([self.remaining_scenes])
        return ret