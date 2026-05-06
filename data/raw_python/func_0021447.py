def parse(self):
        '''Iterate over the lines and extract the required data.'''
        for self.line in self.output:
            # Parse general data: charge, multiplicity, coordinates, etc.
            self.index = 0

            if self.line[1:13] == 'Total Charge':
                tokens = self.line.split()
                self.charge = int(tokens[-1])

            if (self.line[1:13] or self.line[0:12]) == 'Multiplicity':
                tokens = self.line.split()
                self.multiplicity = int(tokens[-1])

            if self.line[0:33] == 'CARTESIAN COORDINATES (ANGSTROEM)':
                if not hasattr(self, 'names'):
                    self.names = dict()
                if not hasattr(self, 'coords'):
                    self.coords = dict()
                self.line = self._skip_lines(2)
                names = list()
                coords = list()
                while self.line.strip():
                    tokens = self.line.split()
                    names.append(tokens[0])
                    x = float(tokens[1])
                    y = float(tokens[2])
                    z = float(tokens[3])
                    coords.append((x, y, z))
                    self.line = next(self.output)
                self.names = np.array(names)
                self.coords[self.index] = np.array(coords)

            if self.line[22:50] == 'MULLIKEN POPULATION ANALYSIS':
                if not hasattr(self, 'populations'):
                    self.populations = dict()
                self.line = self._skip_lines(6)
                populations = list()
                while self.line.strip() and 'Sum' not in self.line:
                    tokens = self.line.split()
                    populations.append((float(tokens[-2]), float(tokens[-1])))
                    self.line = next(self.output)
                self.populations['mulliken'][self.index] = np.array(populations) # noqa

            # Parse data from the EPR/NMR module
            if self.line[37:44] == 'EPR/NMR':
                self.eprnmr = dict()

            if self.line[0:19] == 'ELECTRONIC G-MATRIX':
                self.line = self._skip_lines(4)
                self.eprnmr['g']['tensor'] = self._parse_tensor()

            if self.line[0:27] == 'ZERO-FIELD-SPLITTING TENSOR':
                self.line = self._skip_lines(4)
                self.eprnmr['zfs']['tensor'] = self._parse_tensor()

            if self.line[1:8] == 'Nucleus':
                tokens = self.line.split()
                nucleus = int(re.findall(r'\d+', tokens[1])[0])
                while 'Raw HFC' not in self.line:
                    self.line = self._skip_lines(1)
                self.line = self._skip_lines(2)
                self.eprnmr['hfc'][nucleus]['tensor'] = self._parse_tensor()
                self.line = self._skip_lines(1)
                self.eprnmr['hfc'][nucleus]['fc'] = self._parse_components()
                self.eprnmr['hfc'][nucleus]['sd'] = self._parse_components()
                self.line = self._skip_lines(1)
                self.eprnmr['hfc'][nucleus]['orb'] = self._parse_components()
                self.eprnmr['hfc'][nucleus]['dia'] = self._parse_components()

            # Parse data from the MRCI module
            if self.line[36:43] == 'M R C I':
                self.mrci = dict()

            if self.line[1:19] == 'SPIN-SPIN COUPLING':
                self.line = self._skip_lines(4)
                self.mrci['zfs']['ssc']['tensor'] = self._parse_tensor()

            if self.line[1:30] == '2ND ORDER SPIN-ORBIT COUPLING':
                while 'Second' not in self.line:
                    self.line = self._skip_lines(1)
                self.line = self._skip_lines(1)
                self.mrci['zfs']['soc']['second_order']['0']['tensor'] = self._parse_tensor() # noqa
                self.line = self._skip_lines(2)
                self.mrci['zfs']['soc']['second_order']['m']['tensor'] = self._parse_tensor() # noqa
                self.line = self._skip_lines(2)
                self.mrci['zfs']['soc']['second_order']['p']['tensor'] = self._parse_tensor() # noqa

            if self.line[1:42] == 'EFFECTIVE HAMILTONIAN SPIN-ORBIT COUPLING':
                self.line = self._skip_lines(4)
                self.mrci['zfs']['soc']['heff']['tensor'] = self._parse_tensor()