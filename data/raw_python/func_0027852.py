def close(self):
        """
        Closes the lvm and vg_t handle. Usually you would never need to use this method
        unless you are doing operations using the ctypes function wrappers in conversion.py

        *Raises:*

        *       HandleError
        """
        if self.handle:
            cl = lvm_vg_close(self.handle)
            if cl != 0:
                raise HandleError("Failed to close VG handle after init check.")
            self.__vgh = None
            self.lvm.close()