def _skip_frame(self):
        """Skip one frame"""
        self._get_line()
        num_atoms = int(self._get_line())
        if self.num_atoms is not None and self.num_atoms != num_atoms:
            raise ValueError("The number of atoms must be the same over the entire file.")
        for i in range(num_atoms+1):
            self._get_line()