def send_messages(self, messages):
        """Send one or more EmailMessage objects.

        Returns:
             int: Number of email messages sent.
        """
        if not messages:
            return
        new_conn_created = self.open()
        if not self.connection:
            # We failed silently on open(). Trying to send would be pointless.
            return
        num_sent = 0
        for message in messages:
            sent = self._send(message)
            if sent:
                num_sent += 1
        if new_conn_created:
            self.close()
        return num_sent