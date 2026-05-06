def _read(self):
        """ Helper function to load the data referenced by this bundle. """
        if self._dask:
            d = da.from_delayed(
                delayed(read_from_bpch, )(
                    self.filename, self.file_position, self.shape,
                    self.dtype, self.endian, use_mmap=self._mmap
                ),
                self.shape, self.dtype
            )
        else:
            d = read_from_bpch(
                    self.filename, self.file_position, self.shape,
                    self.dtype, self.endian, use_mmap=self._mmap
            )

        return d