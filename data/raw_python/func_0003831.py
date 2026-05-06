def dump(self, f):
        """Dump the data structure to a file-like object"""
        # header
        print("PSF", file=f)
        print(file=f)

        # title
        print("      1 !NTITLE", file=f)
        print(self.title, file=f)
        print(file=f)

        # atoms
        print("% 7i !NATOM" % len(self.numbers), file=f)
        if len(self.numbers) > 0:
            for index, (number, atom_type, charge, name, molecule) in enumerate(zip(self.numbers, self.atom_types, self.charges, self.names, self.molecules)):
                atom = periodic[number]
                print("% 7i % 4s % 4i NAME % 6s % 6s % 8.4f % 12.6f 0" % (
                    index + 1,
                    name,
                    molecule + 1,
                    atom.symbol,
                    atom_type,
                    charge,
                    atom.mass/unified,
                ), file=f)
        print(file=f)

        # bonds
        print("% 7i !NBOND" % len(self.bonds), file=f)
        if len(self.bonds) > 0:
            tmp = []
            for bond in self.bonds:
                tmp.extend(bond+1)
                if len(tmp) >= 8:
                    print(" ".join("% 7i" % v for v in tmp[:8]), file=f)
                    tmp = tmp[8:]
            if len(tmp) > 0:
                print(" ".join("% 7i" % v for v in tmp), file=f)
        print(file=f)

        # bends
        print("% 7i !NTHETA" % len(self.bends), file=f)
        if len(self.bends) > 0:
            tmp = []
            for bend in self.bends:
                tmp.extend(bend+1)
                if len(tmp) >= 9:
                    print(" " + (" ".join("% 6i" % v for v in tmp[:9])), file=f)
                    tmp = tmp[9:]
            if len(tmp) > 0:
                print(" " + (" ".join("% 6i" % v for v in tmp)), file=f)
        print(file=f)

        # dihedrals
        print("% 7i !NPHI" % len(self.dihedrals), file=f)
        if len(self.dihedrals) > 0:
            tmp = []
            for dihedral in self.dihedrals:
                tmp.extend(dihedral+1)
                if len(tmp) >= 8:
                    print(" " + (" ".join("% 6i" % v for v in tmp[:8])), file=f)
                    tmp = tmp[8:]
            if len(tmp) > 0:
                print(" " + (" ".join("% 6i" % v for v in tmp)), file=f)
        print(file=f)

        # impropers
        print("% 7i !NIMPHI" % len(self.impropers), file=f)
        if len(self.impropers) > 0:
            tmp = []
            for improper in self.impropers:
                tmp.extend(improper+1)
                if len(tmp) >= 8:
                    print(" " + (" ".join("% 6i" % v for v in tmp[:8])), file=f)
                    tmp = tmp[8:]
            if len(tmp) > 0:
                print(" " + (" ".join("% 6i" % v for v in tmp)), file=f)
        print(file=f)

        # not implemented fields
        print("      0 !NDON", file=f)
        print(file=f)
        print("      0 !NNB", file=f)
        print(file=f)
        print("      0 !NGRP", file=f)
        print(file=f)