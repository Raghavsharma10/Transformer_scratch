def size(self, units="MiB"):
        """
        Returns the logical volume size in the given units. Default units are  MiB.

        *Args:*

        *       units (str):    Unit label ('MiB', 'GiB', etc...). Default is MiB.
        """
        self.open()
        size = lvm_lv_get_size(self.__lvh)
        self.close()
        return size_convert(size, units)