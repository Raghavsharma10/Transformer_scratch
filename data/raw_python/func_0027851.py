def open(self):
        """
        Obtains the lvm and vg_t handle. Usually you would never need to use this method
        unless you are doing operations using the ctypes function wrappers in conversion.py

        *Raises:*

        *       HandleError
        """
        if not self.handle:
            self.lvm.open()
            self.__vgh = lvm_vg_open(self.lvm.handle, self.name, self.mode)
            if not bool(self.__vgh):
                raise HandleError("Failed to initialize VG Handle.")