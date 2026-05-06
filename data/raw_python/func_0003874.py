def _read_frame(self):
        """Read a single frame from the trajectory"""
        self._secfile.get_next("Frame Number")
        frame = ATRJFrame()
        # Read the time and energy
        energy_lines = self._secfile.get_next("Time/Energy")
        energy_words = energy_lines[0].split()
        frame.time = float(energy_words[0])*picosecond
        frame.step = int(energy_words[1])
        frame.total_energy = float(energy_words[2])*kcalmol
        # Read the coordinates
        coord_lines = self._secfile.get_next("Coordinates")
        frame.coordinates = np.zeros((self.num_atoms, 3), float)
        for index, line in enumerate(coord_lines):
            words = line.split()
            frame.coordinates[index, 0] = float(words[1])
            frame.coordinates[index, 1] = float(words[2])
            frame.coordinates[index, 2] = float(words[3])
        frame.coordinates *= angstrom
        # Done
        return frame