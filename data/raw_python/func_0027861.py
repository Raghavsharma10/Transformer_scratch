def is_partial(self):
        """
        Returns True if the VG is partial, False otherwise.
        """
        self.open()
        part = lvm_vg_is_partial(self.handle)
        self.close()
        return bool(part)