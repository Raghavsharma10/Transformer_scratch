def copy(self, h5file=None):
        """Create a copy of the current instance

        This is done by recursively copying the underlying hdf5 data.

        Parameters
        ----------
        h5file: str, h5py.File, h5py.Group, or None
            see `QPImage.__init__`
        """
        h5 = copyh5(self.h5, h5file)
        return QPImage(h5file=h5, h5dtype=self.h5dtype)