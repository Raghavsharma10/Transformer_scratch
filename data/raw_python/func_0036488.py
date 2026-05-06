def connect(self):
        """
        Creates a new KazooClient and establishes a connection.

        Passes the client the `handle_connection_change` method as a callback
        to fire when the Zookeeper connection changes state.
        """
        self.client = client.KazooClient(hosts=",".join(self.hosts))

        self.client.add_listener(self.handle_connection_change)
        self.client.start_async()