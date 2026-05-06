def can_receive_messages(self):
        """Whether tihs communication is ready to receive messages.]

        :rtype: bool

        .. code:: python

            assert not communication.can_receive_messages()
            communication.start()
            assert communication.can_receive_messages()
            communication.stop()
            assert not communication.can_receive_messages()

        """
        with self.lock:
            return not self._state.is_waiting_for_start() and \
                not self._state.is_connection_closed()