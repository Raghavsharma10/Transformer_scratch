def from_file(cls, filename):
        '''Create a cube object by loading data from a file.

           *Arguemnts:*

           filename
                The file to load. It must contain the header with the
                description of the grid and the molecule.
        '''
        with open(filename) as f:
            molecule, origin, axes, nrep, subtitle, nuclear_charges = \
                read_cube_header(f)
            data = np.zeros(tuple(nrep), float)
            tmp = data.ravel()
            counter = 0
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                words = line.split()
                for word in words:
                    tmp[counter] = float(word)
                    counter += 1
        return cls(molecule, origin, axes, nrep, data, subtitle, nuclear_charges)