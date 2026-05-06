def deactivate(self):
        """
        Deactivates the logical volume.

        *Raises:*

        *       HandleError
        """
        self.open()
        d = lvm_lv_deactivate(self.handle)
        self.close()
        if d != 0:
            raise CommitError("Failed to deactivate LV.")