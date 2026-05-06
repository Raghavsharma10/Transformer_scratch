def release_client(self, client):
        """Releases a client object to the pool.

        Args:
            client: Client object.
        """
        if isinstance(client, Client):
            if not self._is_expired_client(client):
                LOG.debug('Client is not expired. Adding back to pool')
                self.__pool.append(client)
            elif client.is_connected():
                LOG.debug('Client is expired and connected. Disconnecting')
                client.disconnect()
        if self.__sem is not None:
            self.__sem.release()