def vgscan(self):
        """
        Probes the system for volume groups and returns a list of VolumeGroup
        instances::

            from lvm2py import *

            lvm = LVM()
            vgs = lvm.vgscan()

        *Raises:*

        *       HandleError
        """
        vg_list = []
        self.open()
        names = lvm_list_vg_names(self.handle)
        if not bool(names):
            return vg_list
        vgnames = []
        vg = dm_list_first(names)
        while vg:
            c = cast(vg, POINTER(lvm_str_list))
            vgnames.append(c.contents.str)
            if dm_list_end(names, vg):
                # end of linked list
                break
            vg = dm_list_next(names, vg)
        self.close()
        for name in vgnames:
            vginst = self.get_vg(name)
            vg_list.append(vginst)
        return vg_list