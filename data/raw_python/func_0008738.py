def _sanitise(self):
        """
        Convert attributes of type npumpy.float32 to numpy.float64 so that they will print properly.
        """
        for k in self.__dict__:
            if isinstance(self.__dict__[k], np.float32):  # np.float32 has a broken __str__ method
                self.__dict__[k] = np.float64(self.__dict__[k])