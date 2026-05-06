def dump_children(self, f, indent=''):
        """Dump the children of the current section to a file-like object"""
        for child in self.__order:
            child.dump(f, indent+'  ')