def write_to_file(self, f, file_unit=angstrom):
        """Write the trajectory to a file

           Argument:
            | ``f``  -- a filename or a file-like object to write to

           Optional argument:
            | ``file_unit``  --  the unit of the values written to file
                                 [default=angstrom]
        """
        xyz_writer = XYZWriter(f, self.symbols, file_unit=file_unit)
        for title, coordinates in zip(self.titles, self.geometries):
            xyz_writer.dump(title, coordinates)