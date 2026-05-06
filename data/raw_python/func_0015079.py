def _is_msg(self, msg):
        """
        Returns True if message is a proper Scratch message, else return False.
        """
        if not msg or len(msg) < self.prefix_len:
            return False
        length = self._extract_len(msg[:self.prefix_len])
        msg_type = msg[self.prefix_len:].split(' ', 1)[0]
        if length == len(msg[self.prefix_len:]) and msg_type in self.msg_types:
            return True
        return False