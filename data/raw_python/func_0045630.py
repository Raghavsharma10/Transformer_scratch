def load(self, name):
        """Loads and returns foreign library."""
        name = ctypes.util.find_library(name)
        return ctypes.cdll.LoadLibrary(name)