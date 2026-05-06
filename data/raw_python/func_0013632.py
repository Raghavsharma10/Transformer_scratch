def readWiggleLine(self, line):
        """
        Read a wiggle line. If it is a data line, add values to the
        protocol object.
        """
        if(line.isspace() or line.startswith("#")
                or line.startswith("browser") or line.startswith("track")):
            return
        elif line.startswith("variableStep"):
            self._mode = self._VARIABLE_STEP
            self.parseStep(line)
            return
        elif line.startswith("fixedStep"):
            self._mode = self._FIXED_STEP
            self.parseStep(line)
            return
        elif self._mode is None:
            raise ValueError("Unexpected input line: %s" % line.strip())

        if self._queryReference != self._reference:
            return

        # read data lines
        fields = line.split()
        if self._mode == self._VARIABLE_STEP:
            start = int(fields[0])-1  # to 0-based
            val = float(fields[1])
        else:
            start = self._start
            self._start += self._step
            val = float(fields[0])

        if start < self._queryEnd and start > self._queryStart:
            if self._position is None:
                self._position = start
                self._data.start = start

            # fill gap
            while self._position < start:
                self._data.values.append(float('NaN'))
                self._position += 1
            for _ in xrange(self._span):
                self._data.values.append(val)
            self._position += self._span