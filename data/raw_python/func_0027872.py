def remove_lv(self, lv):
        """
        Removes a logical volume from the volume group::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")
            lv = vg.lvscan()[0]
            vg.remove_lv(lv)

        *Args:*

        *       lv (obj):       A LogicalVolume instance.

        *Raises:*

        *       HandleError,  CommitError, ValueError

        .. note::

            The VolumeGroup instance must be in write mode, otherwise CommitError
            is raised.
        """
        lv.open()
        rm = lvm_vg_remove_lv(lv.handle)
        lv.close()
        if rm != 0:
            raise CommitError("Failed to remove LV.")