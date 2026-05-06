def max_pv_count(self):
        """
        Returns the maximum allowed physical volume count.
        """
        self.open()
        count = lvm_vg_get_max_pv(self.handle)
        self.close()
        return count