def lvscan(self):
        """
        Probes the volume group for logical volumes and returns a list of
        LogicalVolume instances::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg")
            lvs = vg.lvscan()

        *Raises:*

        *       HandleError
        """
        self.open()
        lv_list = []
        lv_handles = lvm_vg_list_lvs(self.handle)
        if not bool(lv_handles):
            return lv_list
        lvh = dm_list_first(lv_handles)
        while lvh:
            c = cast(lvh, POINTER(lvm_lv_list))
            lv = LogicalVolume(self, lvh=c.contents.lv)
            lv_list.append(lv)
            if dm_list_end(lv_handles, lvh):
                # end of linked list
                break
            lvh = dm_list_next(lv_handles, lvh)
        self.close()
        return lv_list