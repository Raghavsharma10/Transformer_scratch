def get_connection(self):
        """
        Return a connection from the pool using the `ConnectionSelector`
        instance.

        It tries to resurrect eligible connections, forces a resurrection when
        no connections are availible and passes the list of live connections to
        the selector instance to choose from.

        Returns a connection instance and it's current fail count.
        """
        self.resurrect()

        # no live nodes, resurrect one by force
        if not self.connections:
            self.resurrect(True)
        connection = self.selector.select(self.connections)

        return connection