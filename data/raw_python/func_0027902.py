def remove_vg(self, vg):
        """
        Removes a volume group::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")
            lvm.remove_vg(vg)

        *Args:*

        *       vg (obj):       A VolumeGroup instance.

        *Raises:*

        *       HandleError,  CommitError

        .. note::

            The VolumeGroup instance must be in write mode, otherwise CommitError
            is raised.
        """
        vg.open()
        rm = lvm_vg_remove(vg.handle)
        if rm != 0:
            vg.close()
            raise CommitError("Failed to remove VG.")
        com = lvm_vg_write(vg.handle)
        if com != 0:
            vg.close()
            raise CommitError("Failed to commit changes to disk.")
        vg.close()