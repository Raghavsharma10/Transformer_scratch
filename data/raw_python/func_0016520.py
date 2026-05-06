def receive(self, message_data, message):
        """This method is called when a new message is received by
        running :meth:`wait`, :meth:`process_next` or :meth:`iterqueue`.

        When a message is received, it passes the message on to the
        callbacks listed in the :attr:`callbacks` attribute.
        You can register callbacks using :meth:`register_callback`.

        :param message_data: The deserialized message data.

        :param message: The :class:`carrot.backends.base.BaseMessage` instance.

        :raises NotImplementedError: If no callbacks has been registered.

        """
        if not self.callbacks:
            raise NotImplementedError("No consumer callbacks registered")
        for callback in self.callbacks:
            callback(message_data, message)