def read(self, ncfile, timegrid_data) -> None:
        """Read the data from the given NetCDF file.

        The argument `timegrid_data` defines the data period of the
        given NetCDF file.

        See the general documentation on class |NetCDFVariableFlat|
        for some examples.
        """
        array = query_array(ncfile, self.name)
        idxs: Tuple[Any] = (slice(None),)
        subdev2index = self.query_subdevice2index(ncfile)
        for devicename, seq in self.sequences.items():
            if seq.NDIM:
                if self._timeaxis:
                    subshape = (array.shape[1],) + seq.shape
                else:
                    subshape = (array.shape[0],) + seq.shape
                subarray = numpy.empty(subshape)
                temp = devicename + '_'
                for prod in self._product(seq.shape):
                    station = temp + '_'.join(str(idx) for idx in prod)
                    idx0 = subdev2index.get_index(station)
                    subarray[idxs+prod] = array[self.get_timeplaceslice(idx0)]
            else:
                idx = subdev2index.get_index(devicename)
                subarray = array[self.get_timeplaceslice(idx)]
            seq.series = seq.adjust_series(timegrid_data, subarray)