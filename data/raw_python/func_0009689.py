def refresh_from_server(self):
        """Refresh the group from the server in place."""
        group = self.manager.get(id=self.id)
        self.__init__(self.manager, **group.data)