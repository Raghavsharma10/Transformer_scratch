def max_lv_count(self):
        """
        Returns the maximum allowed logical volume count.
        """
        self.open()
        count = lvm_vg_get_max_lv(self.handle)
        self.close()
        return count