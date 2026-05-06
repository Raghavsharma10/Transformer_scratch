def extent_count(self):
        """
        Returns the volume group extent count.
        """
        self.open()
        count = lvm_vg_get_extent_count(self.handle)
        self.close()
        return count