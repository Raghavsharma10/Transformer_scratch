def pad(self, val):
        """
        :param val:
        :rtype: bytes
        """
        padding = len(int_to_bytes(self._prime))
        padded = int_to_bytes(val).rjust(padding, b'\x00')
        return padded