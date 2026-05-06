def prnt(self):
        """
        Prints DB data representation of the object.
        """
        print("= = = =\n\n%s object key: \033[32m%s\033[0m" % (self.__class__.__name__, self.key))
        pprnt(self._data or self.clean_value())