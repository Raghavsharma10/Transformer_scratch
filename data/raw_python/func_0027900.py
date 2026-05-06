def get_vg(self, name, mode="r"):
        """
        Returns an instance of VolumeGroup. The name parameter should be an existing
        volume group. By default, all volume groups are open in "read" mode::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg")

        To open a volume group with write permissions set the mode parameter to "w"::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")

        *Args:*

        *       name (str):     An existing volume group name.
        *       mode (str):     "r" or "w" for read/write respectively. Default is "r".

        *Raises:*

        *       HandleError
        """
        vg = VolumeGroup(self, name=name, mode=mode)
        return vg