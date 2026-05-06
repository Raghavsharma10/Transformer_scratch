def free_size(self, units="MiB"):
        """
        Returns the volume group free size in the given units. Default units are  MiB.

        *Args:*

        *       units (str):    Unit label ('MiB', 'GiB', etc...). Default is MiB.
        """
        self.open()
        size = lvm_vg_get_free_size(self.handle)
        self.close()
        return size_convert(size, units)