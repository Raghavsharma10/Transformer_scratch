def set_extent_size(self, length, units):
        """
        Sets the volume group extent size in the given units::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")
            vg.set_extent_size(2, "MiB")

        *Args:*

        *       length (int):   The desired length size.
        *       units (str):    The desired units ("MiB", "GiB", etc...).

        *Raises:*

        *       HandleError,  CommitError, KeyError

        .. note::

            The VolumeGroup instance must be in write mode, otherwise CommitError
            is raised.
        """
        size = length * size_units[units]
        self.open()
        ext = lvm_vg_set_extent_size(self.handle, c_ulong(size))
        self._commit()
        self.close()
        if ext != 0:
            raise CommitError("Failed to set extent size.")