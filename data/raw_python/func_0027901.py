def create_vg(self, name, devices):
        """
        Returns a new instance of VolumeGroup with the given name and added physycal
        volumes (devices)::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.create_vg("myvg", ["/dev/sdb1", "/dev/sdb2"])

        *Args:*

        *       name (str):             A volume group name.
        *       devices (list):         A list of device paths.

        *Raises:*

        *       HandleError, CommitError, ValueError
        """
        self.open()
        vgh = lvm_vg_create(self.handle, name)
        if not bool(vgh):
            self.close()
            raise HandleError("Failed to create VG.")
        for device in devices:
            if not os.path.exists(device):
                self._destroy_vg(vgh)
                raise ValueError("%s does not exist." % device)
            ext = lvm_vg_extend(vgh, device)
            if ext != 0:
                self._destroy_vg(vgh)
                raise CommitError("Failed to extend Volume Group.")
            try:
                self._commit_vg(vgh)
            except CommitError:
                self._destroy_vg(vgh)
                raise CommitError("Failed to add %s to VolumeGroup." % device)
        self._close_vg(vgh)
        vg = VolumeGroup(self, name)
        return vg