def open(self):
        """
        Obtains the lvm, vg_t and pv_t handle. Usually you would never need to use this
        method unless you are doing operations using the ctypes function wrappers in
        conversion.py

        *Raises:*

        *       HandleError
        """
        self.vg.open()
        self.__pvh = lvm_pv_from_uuid(self.vg.handle, self.uuid)
        if not bool(self.__pvh):
            raise HandleError("Failed to initialize PV Handle.")