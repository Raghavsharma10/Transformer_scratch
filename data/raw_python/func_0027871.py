def create_lv(self, name, length, units):
        """
        Creates a logical volume and returns the LogicalVolume instance associated with
        the lv_t handle::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")
            lv = vg.create_lv("mylv", 40, "MiB")

        *Args:*

        *       name (str):             The desired logical volume name.
        *       length (int):           The desired size.
        *       units (str):            The size units.

        *Raises:*

        *       HandleError,  CommitError, ValueError

        .. note::

            The VolumeGroup instance must be in write mode, otherwise CommitError
            is raised.
        """
        if units != "%":
            size = size_units[units] * length
        else:
            if not (0 < length <= 100) or type(length) is float:
                raise ValueError("Length not supported.")
            size = (self.size("B") / 100) * length
        self.open()
        lvh = lvm_vg_create_lv_linear(self.handle, name, c_ulonglong(size))
        if not bool(lvh):
            self.close()
            raise CommitError("Failed to create LV.")
        lv = LogicalVolume(self, lvh=lvh)
        self.close()
        return lv