def is_suspended(self):
        """
        Returns True if the logical volume is suspended, False otherwise.
        """
        self.open()
        susp = lvm_lv_is_suspended(self.__lvh)
        self.close()
        return bool(susp)