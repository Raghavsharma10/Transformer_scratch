def powerUp(self, powerup, interface=None, priority=0):
        """
        Installs a powerup (e.g. plugin) on an item or store.

        Powerups will be returned in an iterator when queried for using the
        'powerupsFor' method.  Normally they will be returned in order of
        installation [this may change in future versions, so please don't
        depend on it].  Higher priorities are returned first.  If you have
        something that should run before "normal" powerups, pass
        POWERUP_BEFORE; if you have something that should run after, pass
        POWERUP_AFTER.  We suggest not depending too heavily on order of
        execution of your powerups, but if finer-grained control is necessary
        you may pass any integer.  Normal (unspecified) priority is zero.

        Powerups will only be installed once on a given item.  If you install a
        powerup for a given interface with priority 1, then again with priority
        30, the powerup will be adjusted to priority 30 but future calls to
        powerupFor will still only return that powerup once.


        If no interface or priority are specified, and the class of the
        powerup has a "powerupInterfaces" attribute (containing
        either a sequence of interfaces, or a sequence of
        (interface, priority) tuples), this object will be powered up
        with the powerup object on those interfaces.

        If no interface or priority are specified and the powerup has
        a "__getPowerupInterfaces__" method, it will be called with
        an iterable of (interface, priority) tuples, collected from the
        "powerupInterfaces" attribute described above. The iterable of
        (interface, priority) tuples it returns will then be
        installed.


        @param powerup: an Item that implements C{interface} (if specified)
        @param interface: a zope interface, or None

        @param priority: An int; preferably either POWERUP_BEFORE,
        POWERUP_AFTER, or unspecified.

        @raise TypeError: raises if interface is IPowerupIndirector You may not
        install a powerup for IPowerupIndirector because that would be
        nonsensical.
        """
        if interface is None:
            for iface, priority in powerup._getPowerupInterfaces():
                self.powerUp(powerup, iface, priority)

        elif interface is IPowerupIndirector:
            raise TypeError(
                "You cannot install a powerup for IPowerupIndirector: " +
                powerup)
        else:
            forc = self.store.findOrCreate(_PowerupConnector,
                                           item=self,
                                           interface=unicode(qual(interface)),
                                           powerup=powerup)
            forc.priority = priority