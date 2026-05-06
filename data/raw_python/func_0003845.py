def read_from_file(cls, filename):
        """Construct a MolecularDistortion object from a file"""
        with open(filename) as f:
            lines = list(line for line in f if line[0] != '#')
        r = []
        t = []
        for line in lines[:3]:
            values = list(float(word) for word in line.split())
            r.append(values[:3])
            t.append(values[3])
        transformation = Complete(r, t)
        affected_atoms = set(int(word) for word in lines[3].split())
        return cls(affected_atoms, transformation)