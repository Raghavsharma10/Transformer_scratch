def upgradeBatch(self, n):
        """
        Upgrade the entire store in batches, yielding after each batch.

        @param n: Number of upgrades to perform per transaction
        @type n: C{int}

        @raise axiom.errors.ItemUpgradeError: if an item upgrade failed

        @return: A generator that yields after each batch upgrade. This needs
            to be consumed for upgrading to actually take place.
        """
        store = self.store

        def _doBatch(itemType):
            upgradedAnything = False

            for theItem in store.query(itemType, limit=n):
                upgradedAnything = True
                try:
                    self.upgradeItem(theItem)
                except:
                    f = Failure()
                    raise ItemUpgradeError(
                        f, theItem.storeID, itemType,
                        _typeNameToMostRecentClass[itemType.typeName])

            return upgradedAnything

        if self.upgradesPending:
            didAny = False

            while self._oldTypesRemaining:
                t0 = self._oldTypesRemaining[0]

                upgradedAnything = store.transact(_doBatch, t0)
                if not upgradedAnything:
                    self._oldTypesRemaining.pop(0)
                    if didAny:
                        msg("%s finished upgrading %s" % (store.dbdir.path, qual(t0)))
                    continue
                elif not didAny:
                    didAny = True
                    msg("%s beginning upgrade..." % (store.dbdir.path,))

                yield None

            if didAny:
                msg("%s completely upgraded." % (store.dbdir.path,))