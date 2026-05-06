def is_active(self):
        """
        Returns True if the logical volume is active, False otherwise.
        """
        self.open()
        active = lvm_lv_is_active(self.__lvh)
        self.close()
        return bool(active)