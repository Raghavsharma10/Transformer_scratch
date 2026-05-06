def load(self, path):
        """
        Load a GADDAG from file, replacing the words currently in this GADDAG.

        Args:
            path: path to saved GADDAG to be loaded.
        """
        path = os.path.expandvars(os.path.expanduser(path))

        gdg = cgaddag.gdg_load(path.encode("ascii"))
        if not gdg:
            errno = ctypes.c_int.in_dll(ctypes.pythonapi, "errno").value
            raise OSError(errno, os.strerror(errno), path)

        self.__del__()
        self.gdg = gdg.contents