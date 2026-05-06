def pump_reader(self):
        """
        Synchronously reads one message from the watch, blocking until a message is available.
        All events caused by the message read will be processed before this method returns.

        .. note::
           You usually don't need to invoke this method manually; instead, see :meth:`run_sync` and :meth:`run_async`.
        """
        origin, message = self.transport.read_packet()
        if isinstance(origin, MessageTargetWatch):
            self._handle_watch_message(message)
        else:
            self._broadcast_transport_message(origin, message)