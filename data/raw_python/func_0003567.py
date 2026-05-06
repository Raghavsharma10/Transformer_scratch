def send(self, host_message_class, *args):
        """Send a host message.

        :param type host_message_class: a subclass of
          :class:`AYABImterface.communication.host_messages.Message`
        :param args: additional arguments that shall be passed to the
          :paramref:`host_message_class` as arguments
        """
        message = host_message_class(self._file, self, *args)
        with self.lock:
            message.send()
            for callable in self._on_message:
                callable(message)