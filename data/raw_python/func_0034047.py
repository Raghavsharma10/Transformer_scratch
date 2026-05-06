def listen(self):
        """
        Set up a quick connection. Returns on disconnect.

        After calling `connect()`, this waits for messages from the server
        using `select`, and notifies the subscriber of any events.
        """
        import select
        while self.connected:
            r, w, e = select.select((self.ws.sock, ), (), ())
            if r:
                self.on_message()
            elif e:
                self.subscriber.on_sock_error(e)
        self.disconnect()