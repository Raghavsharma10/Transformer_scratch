def add_pv(self, device):
        """
        Initializes a device as a physical volume and adds it to the volume group::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")
            vg.add_pv("/dev/sdbX")

        *Args:*

        *       device (str):   An existing device.

        *Raises:*

        *       ValueError, CommitError, HandleError

       .. note::

            The VolumeGroup instance must be in write mode, otherwise CommitError
            is raised.
        """
        if not os.path.exists(device):
            raise ValueError("%s does not exist." % device)
        self.open()
        ext = lvm_vg_extend(self.handle, device)
        if ext != 0:
            self.close()
            raise CommitError("Failed to extend Volume Group.")
        self._commit()
        self.close()
        return PhysicalVolume(self, name=device)