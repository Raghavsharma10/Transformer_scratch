def powerupsFor(self, interface):
        """
        Returns powerups installed using C{powerUp}, in order of descending
        priority.

        Powerups found to have been deleted, either during the course of this
        powerupsFor iteration, during an upgrader, or previously, will not be
        returned.
        """
        inMemoryPowerup = self._inMemoryPowerups.get(interface, None)
        if inMemoryPowerup is not None:
            yield inMemoryPowerup
        if self.store is None:
            return
        name = unicode(qual(interface), 'ascii')
        for cable in self.store.query(
            _PowerupConnector,
            AND(_PowerupConnector.interface == name,
                _PowerupConnector.item == self),
            sort=_PowerupConnector.priority.descending):
            pup = cable.powerup
            if pup is None:
                # this powerup was probably deleted during an upgrader.
                cable.deleteFromStore()
            else:
                indirector = IPowerupIndirector(pup, None)
                if indirector is not None:
                    yield indirector.indirect(interface)
                else:
                    yield pup