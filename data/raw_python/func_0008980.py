def preconnect(self, size=-1):
        """(pre)Connects some or all redis clients inside the pool.

        Args:
            size (int): number of redis clients to build and to connect
                (-1 means all clients if pool max_size > -1)

        Raises:
            ClientError: when size == -1 and pool max_size == -1
        """
        if size == -1 and self.max_size == -1:
            raise ClientError("size=-1 not allowed with pool max_size=-1")
        limit = min(size, self.max_size) if size != -1 else self.max_size
        clients = yield [self.get_connected_client() for _ in range(0, limit)]
        for client in clients:
            self.release_client(client)