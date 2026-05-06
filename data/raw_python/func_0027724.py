def name(self):
        """
        Returns the logical volume name.
        """
        self.open()
        name = lvm_lv_get_name(self.__lvh)
        self.close()
        return name