def powerDown(self, powerup, interface=None):
        """
        Remove a powerup.

        If no interface is specified, and the type of the object being
        installed has a "powerupInterfaces" attribute (containing
        either a sequence of interfaces, or a sequence of (interface,
        priority) tuples), the target will be powered down with this
        object on those interfaces.

        If this object has a "__getPowerupInterfaces__" method, it
        will be called with an iterable of (interface, priority)
        tuples. The iterable of (interface, priority) tuples it
        returns will then be uninstalled.

        (Note particularly that if powerups are added or removed to the
        collection described above between calls to powerUp and powerDown, more
        powerups or less will be removed than were installed.)
        """
        if interface is None:
            for interface, priority in powerup._getPowerupInterfaces():
                self.powerDown(powerup, interface)
        else:
            for cable in self.store.query(_PowerupConnector,
                                          AND(_PowerupConnector.item == self,
                                              _PowerupConnector.interface == unicode(qual(interface)),
                                              _PowerupConnector.powerup == powerup)):
                cable.deleteFromStore()
                return
            raise ValueError("Not powered up for %r with %r" % (interface,
                                                                powerup))