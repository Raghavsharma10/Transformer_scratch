def is_exported(self):
        """
        Returns True if the VG is exported, False otherwise.
        """
        self.open()
        exp = lvm_vg_is_exported(self.handle)
        self.close()
        return bool(exp)