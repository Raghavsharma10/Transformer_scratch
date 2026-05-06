def pv_count(self):
        """
        Returns the physical volume count.
        """
        self.open()
        count = lvm_vg_get_pv_count(self.handle)
        self.close()
        return count