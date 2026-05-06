def load_obsdata(self, idx: int) -> None:
        """Load the next obs sequence value (of the given index)."""
        if self._obs_ramflag:
            self.obs[0] = self._obs_array[idx]
        elif self._obs_diskflag:
            raw = self._obs_file.read(8)
            self.obs[0] = struct.unpack('d', raw)