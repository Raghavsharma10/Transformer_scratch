def generate_identity(self):
        """
        Generate a unique but random identity.
        """
        identity = struct.pack('!BI', 0, self._base_identity)
        self._base_identity += 1

        if self._base_identity >= 2 ** 32:
            self._base_identity = 0

        return identity