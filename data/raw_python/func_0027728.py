def activate(self):
        """
        Activates the logical volume.

        *Raises:*

        *       HandleError
        """
        self.open()
        a = lvm_lv_activate(self.handle)
        self.close()
        if a != 0:
            raise CommitError("Failed to activate LV.")