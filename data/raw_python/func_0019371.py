def write(self, ncfile) -> None:
        """Write the data to the given NetCDF file.

        See the general documentation on classes |NetCDFVariableDeep|
        and |NetCDFVariableAgg| for some examples.
        """
        self: NetCDFVariableBase
        self.insert_subdevices(ncfile)
        dimensions = self.dimensions
        array = self.array
        for dimension, length in zip(dimensions[2:], array.shape[2:]):
            create_dimension(ncfile, dimension, length)
        create_variable(ncfile, self.name, 'f8', dimensions)
        ncfile[self.name][:] = array