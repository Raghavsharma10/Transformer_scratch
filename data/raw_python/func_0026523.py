def client_disconnect(self, event):
        """
        A client has disconnected, update possible subscriptions accordingly.

        :param event:
        """
        self.log("Removing disconnected client from subscriptions", lvl=debug)
        client_uuid = event.clientuuid
        self._unsubscribe(client_uuid)