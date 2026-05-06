def pvscan(self):
        """
        Probes the volume group for physical volumes and returns a list of
        PhysicalVolume instances::

            from lvm2py import *

            lvm = LVM()
            vg = lvm.get_vg("myvg")
            pvs = vg.pvscan()

        *Raises:*

        *       HandleError
        """
        self.open()
        pv_list = []
        pv_handles = lvm_vg_list_pvs(self.handle)
        if not bool(pv_handles):
            return pv_list
        pvh = dm_list_first(pv_handles)
        while pvh:
            c = cast(pvh, POINTER(lvm_pv_list))
            pv = PhysicalVolume(self, pvh=c.contents.pv)
            pv_list.append(pv)
            if dm_list_end(pv_handles, pvh):
                # end of linked list
                break
            pvh = dm_list_next(pv_handles, pvh)
        self.close()
        return pv_list