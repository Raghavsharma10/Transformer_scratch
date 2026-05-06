def read(self, ncfile, timegrid_data) -> None:
        """Read the data from the given NetCDF file.

        The argument `timegrid_data` defines the data period of the
        given NetCDF file.

        See the general documentation on class |NetCDFVariableDeep|
        for some examples.
        """
        array = query_array(ncfile, self.name)
        subdev2index = self.query_subdevice2index(ncfile)
        for subdevice, sequence in self.sequences.items():
            idx = subdev2index.get_index(subdevice)
            values = array[self.get_slices(idx, sequence.shape)]
            sequence.series = sequence.adjust_series(
                timegrid_data, values)