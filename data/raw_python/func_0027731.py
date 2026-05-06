def name(self):
        """
        Returns the physical volume device path.
        """
        self.open()
        name = lvm_pv_get_name(self.handle)
        self.close()
        return name