def _read_frame(self):
        """Read one frame"""
        # Read the first line, ignore the title and try to get the time. The
        # time field is optional.
        line = self._get_line()
        pos = line.rfind("t=")
        if pos >= 0:
            time = float(line[pos+2:])*picosecond
        else:
            time = 0.0
        # Read the second line, the number of atoms must match with the first
        # frame.
        num_atoms = int(self._get_line())
        if self.num_atoms is not None and self.num_atoms != num_atoms:
            raise ValueError("The number of atoms must be the same over the entire file.")
        # Read the atom lines
        pos = np.zeros((num_atoms, 3), np.float32)
        vel = np.zeros((num_atoms, 3), np.float32)
        for i in range(num_atoms):
            words = self._get_line()[22:].split()
            pos[i, 0] = float(words[0])
            pos[i, 1] = float(words[1])
            pos[i, 2] = float(words[2])
            vel[i, 0] = float(words[3])
            vel[i, 1] = float(words[4])
            vel[i, 2] = float(words[5])
        pos *= nanometer
        vel *= nanometer/picosecond
        # Read the cell line
        cell = np.zeros((3, 3), np.float32)
        words = self._get_line().split()
        if len(words) >= 3:
            cell[0, 0] = float(words[0])
            cell[1, 1] = float(words[1])
            cell[2, 2] = float(words[2])
        if len(words) == 9:
            cell[1, 0] = float(words[3])
            cell[2, 0] = float(words[4])
            cell[0, 1] = float(words[5])
            cell[2, 1] = float(words[6])
            cell[0, 2] = float(words[7])
            cell[1, 2] = float(words[8])
        cell *= nanometer
        return time, pos, vel, cell