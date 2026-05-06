def write(self, ncfile) -> None:
        """Write the data to the given NetCDF file.

        See the general documentation on class |NetCDFVariableFlat|
        for some examples.
        """
        self.insert_subdevices(ncfile)
        create_variable(ncfile, self.name, 'f8', self.dimensions)
        ncfile[self.name][:] = self.array