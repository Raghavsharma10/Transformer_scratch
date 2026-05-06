def get_pv(self, device):
        """
        Returns the physical volume associated with the given device::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg", "w")
            vg.get_pv("/dev/sdb1")

        *Args:*

        *       device (str):   An existing device.

        *Raises:*

        *       ValueError, HandleError
        """
        if not os.path.exists(device):
            raise ValueError("%s does not exist." % device)
        return PhysicalVolume(self, name=device)