def interfacesFor(self, powerup):
        """
        Return an iterator of the interfaces for which the given powerup is
        installed on this object.

        This is not implemented for in-memory powerups.  It will probably fail
        in an unpredictable, implementation-dependent way if used on one.
        """
        pc = _PowerupConnector
        for iface in self.store.query(pc,
                                      AND(pc.item == self,
                                          pc.powerup == powerup)).getColumn('interface'):
            yield namedAny(iface)