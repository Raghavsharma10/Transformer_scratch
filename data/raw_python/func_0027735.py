def free(self, units="MiB"):
        """
        Returns the free size in the given units. Default units are  MiB.

        *Args:*

        *       units (str):    Unit label ('MiB', 'GiB', etc...). Default is MiB.
        """
        self.open()
        size = lvm_pv_get_free(self.handle)
        self.close()
        return size_convert(size, units)