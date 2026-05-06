def uninstall(self):
        """
        Called when uninstalled from the user store. Uninstalls all my
        powerups.
        """
        for item in self.items:
            uninstallFrom(item, self.store)
        self._items = []