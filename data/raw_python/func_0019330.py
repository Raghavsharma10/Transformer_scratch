def load_simdata(self, idx: int) -> None:
        """Load the next sim sequence value (of the given index)."""
        if self._sim_ramflag:
            self.sim[0] = self._sim_array[idx]
        elif self._sim_diskflag:
            raw = self._sim_file.read(8)
            self.sim[0] = struct.unpack('d', raw)