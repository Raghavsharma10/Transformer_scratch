def write(self, metadata, payload):
        """Write metadata

        metadata is string:string dict.
        payload must be encoded as string.
        """
        a = self.get_active_archive()
        a.write(metadata, payload)
        if self._should_roll_archive():
            self._roll_archive()