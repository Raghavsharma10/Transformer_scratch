def remove_pv(self, pv):
        """
        Removes a physical volume from the volume group::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")
            pv = vg.pvscan()[0]
            vg.remove_pv(pv)

        *Args:*

        *       pv (obj):       A PhysicalVolume instance.

        *Raises:*

        *       HandleError, CommitError

        .. note::

            The VolumeGroup instance must be in write mode, otherwise CommitError
            is raised. Also, when removing the last physical volume, the volume
            group is deleted in lvm, leaving the instance with a null handle.
        """
        name = pv.name
        self.open()
        rm = lvm_vg_reduce(self.handle, name)
        if rm != 0:
            self.close()
            raise CommitError("Failed to remove %s." % name)
        self._commit()
        self.close()