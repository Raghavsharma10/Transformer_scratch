def send(self):
        """
        Sends the object to the watch. Block until completion, or raises :exc:`.PutBytesError` on failure.

        During transmission, a "progress" event will be periodically emitted with the following signature: ::

           (sent_this_interval, sent_so_far, total_object_size)
        """
        # Prepare the watch to receive something.
        cookie = self._prepare()

        # Send it.
        self._send_object(cookie)

        # Commit it.
        self._commit(cookie)

        # Install it.
        self._install(cookie)