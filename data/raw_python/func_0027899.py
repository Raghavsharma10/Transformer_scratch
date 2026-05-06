def close(self):
        """
        Closes the lvm handle. Usually you would never need to use this method unless
        you are trying to do operations using the ctypes function wrappers in conversion.py

        *Raises:*

        *       HandleError
        """
        if self.handle:
            q = lvm_quit(self.handle)
            if q != 0:
                raise HandleError("Failed to close LVM handle.")
            self.__handle = None