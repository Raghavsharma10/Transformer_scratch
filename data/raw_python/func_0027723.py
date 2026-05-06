def open(self):
        """
        Obtains the lvm, vg_t and lv_t handle. Usually you would never need to use this
        method unless you are doing operations using the ctypes function wrappers in
        conversion.py

        *Raises:*

        *       HandleError
        """
        self.vg.open()
        self.__lvh = lvm_lv_from_uuid(self.vg.handle, self.uuid)
        if not bool(self.__lvh):
            raise HandleError("Failed to initialize LV Handle.")