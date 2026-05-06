def from_file(cls, filename):
        """Construct a molecule object read from the given file.

           The file format is inferred from the extensions. Currently supported
           formats are: ``*.cml``, ``*.fchk``, ``*.pdb``, ``*.sdf``, ``*.xyz``

           If a file contains more than one molecule, only the first one is
           read.

           Argument:
            | ``filename``  --  the name of the file containing the molecule

           Example usage::

             >>> mol = Molecule.from_file("foo.xyz")
        """
        # TODO: many different API's to load files. brrr...
        if filename.endswith(".cml"):
            from molmod.io import load_cml
            return load_cml(filename)[0]
        elif filename.endswith(".fchk"):
            from molmod.io import FCHKFile
            fchk = FCHKFile(filename, field_labels=[])
            return fchk.molecule
        elif filename.endswith(".pdb"):
            from molmod.io import load_pdb
            return load_pdb(filename)
        elif filename.endswith(".sdf"):
            from molmod.io import SDFReader
            return next(SDFReader(filename))
        elif filename.endswith(".xyz"):
            from molmod.io import XYZReader
            xyz_reader = XYZReader(filename)
            title, coordinates = next(xyz_reader)
            return Molecule(xyz_reader.numbers, coordinates, title, symbols=xyz_reader.symbols)
        else:
            raise ValueError("Could not determine file format for %s." % filename)