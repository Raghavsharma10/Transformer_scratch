def read_from_file(self, filename):
        """Load a PSF file"""
        self.clear()
        with open(filename) as f:
            # A) check the first line
            line = next(f)
            if not line.startswith("PSF"):
                raise FileFormatError("Error while reading: A PSF file must start with a line 'PSF'.")
            # B) read in all the sections, without interpreting them
            current_section = None
            sections = {}
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                elif "!N" in line:
                    words = line.split()
                    current_section = []
                    section_name = words[1][2:]
                    if section_name.endswith(":"):
                        section_name = section_name[:-1]
                    sections[section_name] = current_section
                else:
                    current_section.append(line)
        # C) interpret the supported sections
        # C.1) The title
        self.title = sections['TITLE'][0]
        molecules = []
        numbers = []
        # C.2) The atoms and molecules
        for line in sections['ATOM']:
            words = line.split()
            self.atom_types.append(words[5])
            self.charges.append(float(words[6]))
            self.names.append(words[3])
            molecules.append(int(words[2]))
            atom = periodic[words[4]]
            if atom is None:
                numbers.append(0)
            else:
                numbers.append(periodic[words[4]].number)
        self.molecules = np.array(molecules)-1
        self.numbers = np.array(numbers)
        self.charges = np.array(self.charges)
        # C.3) The bonds section
        tmp = []
        for line in sections['BOND']:
            tmp.extend(int(word) for word in line.split())
        self.bonds = np.reshape(np.array(tmp), (-1, 2))-1
        # C.4) The bends section
        tmp = []
        for line in sections['THETA']:
            tmp.extend(int(word) for word in line.split())
        self.bends = np.reshape(np.array(tmp), (-1, 3))-1
        # C.5) The dihedral section
        tmp = []
        for line in sections['PHI']:
            tmp.extend(int(word) for word in line.split())
        self.dihedrals = np.reshape(np.array(tmp), (-1, 4))-1
        # C.6) The improper section
        tmp = []
        for line in sections['IMPHI']:
            tmp.extend(int(word) for word in line.split())
        self.impropers = np.reshape(np.array(tmp), (-1, 4))-1