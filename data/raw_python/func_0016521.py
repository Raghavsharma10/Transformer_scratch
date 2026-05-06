def discard_all(self, filterfunc=None):
        """Discard all waiting messages.

        :param filterfunc: A filter function to only discard the messages this
            filter returns.

        :returns: the number of messages discarded.

        *WARNING*: All incoming messages will be ignored and not processed.

        Example using filter:

            >>> def waiting_feeds_only(message):
            ...     try:
            ...         message_data = message.decode()
            ...     except: # Should probably be more specific.
            ...         pass
            ...
            ...     if message_data.get("type") == "feed":
            ...         return True
            ...     else:
            ...         return False
        """
        if not filterfunc:
            return self.backend.queue_purge(self.queue)

        if self.no_ack or self.auto_ack:
            raise Exception("discard_all: Can't use filter with auto/no-ack.")

        discarded_count = 0
        while True:
            message = self.fetch()
            if message is None:
                return discarded_count

            if filterfunc(message):
                message.ack()
                discarded_count += 1