def _read_frame(self):
        """Read and return the next time frame"""
        pos = np.zeros((self.num_atoms, 3), float)
        vel = np.zeros((self.num_atoms, 3), float)
        for i in range(self.num_atoms):
            line = next(self._f)
            words = line.split()
            pos[i, 0] = float(words[1])
            pos[i, 1] = float(words[2])
            pos[i, 2] = float(words[3])
            vel[i, 0] = float(words[4])
            vel[i, 1] = float(words[5])
            vel[i, 2] = float(words[6])
        return pos, vel