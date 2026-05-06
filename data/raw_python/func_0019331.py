def save_simdata(self, idx: int) -> None:
        """Save the last sim sequence value (of the given index)."""
        if self._sim_ramflag:
            self._sim_array[idx] = self.sim[0]
        elif self._sim_diskflag:
            raw = struct.pack('d', self.sim[0])
            self._sim_file.write(raw)