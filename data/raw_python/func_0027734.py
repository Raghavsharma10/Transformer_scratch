def dev_size(self, units="MiB"):
        """
        Returns the device size in the given units. Default units are  MiB.

        *Args:*

        *       units (str):    Unit label ('MiB', 'GiB', etc...). Default is MiB.
        """
        self.open()
        size = lvm_pv_get_dev_size(self.handle)
        self.close()
        return size_convert(size, units)