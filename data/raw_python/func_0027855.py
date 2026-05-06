def free_extent_count(self):
        """
        Returns the volume group free extent count.
        """
        self.open()
        count = lvm_vg_get_free_extent_count(self.handle)
        self.close()
        return count