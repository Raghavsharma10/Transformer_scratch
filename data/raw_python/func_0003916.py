def _read_frame(self):
        """Read a single frame from the trajectory"""
        # auxiliary read function
        def read_three(msg):
            """Read three words as floating point numbers"""
            line = next(self._f)
            try:
                return [float(line[:12]), float(line[12:24]), float(line[24:])]
            except ValueError:
                raise FileFormatError(msg)

        frame = {}
        # read the frame header line
        words = next(self._f).split()
        if len(words) != 6:
            raise FileFormatError("The first line of each time frame must contain 6 words. (%i'th frame)" % self._counter)
        if words[0] != "timestep":
            raise FileFormatError("The first word of the first line of each time frame must be 'timestep'. (%i'th frame)" % self._counter)
        try:
            step = int(words[1])
            frame["step"] = step
            if int(words[2]) != self.num_atoms:
                raise FileFormatError("The number of atoms has changed. (%i'th frame, %i'th step)" % (self._counter, step))
            if int(words[3]) != self.keytrj:
                raise FileFormatError("keytrj has changed. (%i'th frame, %i'th step)" % (self._counter, step))
            if int(words[4]) != self.imcon:
                raise FileFormatError("imcon has changed. (%i'th frame, %i'th step)" % (self._counter, step))
            frame["timestep"] = float(words[5])*self.time_unit
            frame["time"] = frame["timestep"]*step # this is ugly, or wait ... dlpoly is a bit ugly. we are not to blame!
        except ValueError:
            raise FileFormatError("Could not convert all numbers on the first line of the current time frame. (%i'th frame)" % self._counter)
        # the three cell lines
        cell = np.zeros((3, 3), float)
        frame["cell"] = cell
        cell_msg = "The cell lines must consist of three floating point values. (%i'th frame, %i'th step)" % (self._counter, step)
        for i in range(3):
            cell[:, i] = read_three(cell_msg)
        cell *= self.pos_unit
        # the atoms
        symbols = []
        frame["symbols"] = symbols
        masses = np.zeros(self.num_atoms, float)
        frame["masses"] = masses
        charges = np.zeros(self.num_atoms, float)
        frame["charges"] = charges
        pos = np.zeros((self.num_atoms, 3), float)
        frame["pos"] = pos
        if self.keytrj > 0:
            vel = np.zeros((self.num_atoms, 3), float)
            frame["vel"] = vel
        if self.keytrj > 1:
            frc = np.zeros((self.num_atoms, 3), float)
            frame["frc"] = frc
        for i in range(self.num_atoms):
            # the atom header line
            words = next(self._f).split()
            if len(words) != 4:
                raise FileFormatError("The atom header line must contain 4 words. (%i'th frame, %i'th step, %i'th atom)" % (self._counter, step, i+1))
            symbols.append(words[0])
            try:
                masses[i] = float(words[2])*self.mass_unit
                charges[i] = float(words[3])
            except ValueError:
                raise FileFormatError("The numbers in the atom header line could not be interpreted.")
            # the pos line
            pos_msg = "The position lines must consist of three floating point values. (%i'th frame, %i'th step, %i'th atom)" % (self._counter, step, i+1)
            pos[i] = read_three(pos_msg)
            if self.keytrj > 0:
                vel_msg = "The velocity lines must consist of three floating point values. (%i'th frame, %i'th step, %i'th atom)" % (self._counter, step, i+1)
                vel[i] = read_three(vel_msg)
            if self.keytrj > 1:
                frc_msg = "The force lines must consist of three floating point values. (%i'th frame, %i'th step, %i'th atom)" % (self._counter, step, i+1)
                frc[i] = read_three(frc_msg)
        pos *= self.pos_unit # convert to au
        if self.keytrj > 0:
            vel *= self.vel_unit # convert to au
        if self.keytrj > 1:
            frc *= self.frc_unit # convert to au
        return frame