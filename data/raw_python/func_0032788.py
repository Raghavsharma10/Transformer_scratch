def sharedInterfaces():
        """
        This attribute is the public interface for code which wishes to discover
        the list of interfaces allowed by this Share.  It is a list of
        Interface objects.
        """
        def get(self):
            if not self.sharedInterfaceNames:
                return ()
            if self.sharedInterfaceNames == ALL_IMPLEMENTED_DB:
                I = implementedBy(self.sharedItem.__class__)
                L = list(I)
                T = tuple(L)
                return T
            else:
                return tuple(map(namedAny, self.sharedInterfaceNames.split(u',')))
        def set(self, newValue):
            self.sharedAttributeNames = _interfacesToNames(newValue)
        return get, set