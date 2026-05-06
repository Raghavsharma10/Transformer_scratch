def get(self, name):
        """Returns a Notification by name.
        """
        if not self.loaded:
            raise RegistryNotLoaded(self)
        if not self._registry.get(name):
            raise NotificationNotRegistered(
                f"Notification not registered. Got '{name}'."
            )
        return self._registry.get(name)