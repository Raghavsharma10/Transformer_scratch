def install(self):
        """
        Called when installed on the user store. Installs my powerups.
        """
        items = []
        for typeName in self.types:
            it = self.store.findOrCreate(namedAny(typeName))
            installOn(it, self.store)
            items.append(str(it.storeID).decode('ascii'))
        self._items = items