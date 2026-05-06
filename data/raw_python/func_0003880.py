def append(self, child):
        """Add a child section or keyword"""
        if not (isinstance(child, CP2KSection) or isinstance(child, CP2KKeyword)):
            raise TypeError("The child must be a CP2KSection or a CP2KKeyword, got: %s." % child)
        l = self.__index.setdefault(child.name, [])
        l.append(child)
        self.__order.append(child)