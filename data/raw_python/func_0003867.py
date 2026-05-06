def write_to_file(self, filename):
        """Write the molecular geometry to a file.

           The file format is inferred from the extensions. Currently supported
           formats are: ``*.xyz``, ``*.cml``

           Argument:
            | ``filename``  --  a filename
        """
        # TODO: give all file format writers the same API
        if filename.endswith('.cml'):
            from molmod.io import dump_cml
            dump_cml(filename, [self])
        elif filename.endswith('.xyz'):
            from molmod.io import XYZWriter
            symbols = []
            for n in self.numbers:
                atom = periodic[n]
                if atom is None:
                    symbols.append("X")
                else:
                    symbols.append(atom.symbol)
            xyz_writer = XYZWriter(filename, symbols)
            xyz_writer.dump(self.title, self.coordinates)
            del xyz_writer
        else:
            raise ValueError("Could not determine file format for %s." % filename)