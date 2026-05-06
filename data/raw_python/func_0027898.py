def open(self):
        """
        Obtains the lvm handle. Usually you would never need to use this method unless
        you are trying to do operations using the ctypes function wrappers in conversion.py

        *Raises:*

        *       HandleError
        """
        if not self.handle:
            try:
                path = self.system_dir
            except AttributeError:
                path = ''
            self.__handle = lvm_init(path)
            if not bool(self.__handle):
                raise HandleError("Failed to initialize LVM handle.")