def mda_count(self):
        """
        Returns the physical volume mda count.
        """
        self.open()
        mda = lvm_pv_get_mda_count(self.handle)
        self.close()
        return mda