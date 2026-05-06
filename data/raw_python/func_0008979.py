def destroy(self):
        """Disconnects all pooled client objects."""
        while True:
            try:
                client = self.__pool.popleft()
                if isinstance(client, Client):
                    client.disconnect()
            except IndexError:
                break