def dump(self, f, indent=''):
        """Dump this keyword to a file-like object"""
        if self.__unit is None:
            print(("%s%s %s" % (indent, self.__name, self.__value)).rstrip(), file=f)
        else:
            print(("%s%s [%s] %s" % (indent, self.__name, self.__unit, self.__value)).rstrip(), file=f)